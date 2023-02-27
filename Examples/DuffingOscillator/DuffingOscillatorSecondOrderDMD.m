% This script generates trajectories of the Duffing oscillator and uses
% first order and second order Liouville DMD to generate a predictive model 
% from the data.
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
close all
clear all
addpath('../../lib')
%% Dataset parameters

% Duffing oscillator parameters and model
n = 1; % Output dimension
order = 2; % Order of the system (code only for order = 2)
stateDimension = order*n; % State space dimension
alpha = 1;
beta = -1;
delta = 0;
xDot = @(t,x) [x(2);-delta*x(2)-beta*x(1)-alpha*x(1)^3];

% Initial conditions, 10 grid points per dimension in a 10 x 10 square 
% centered at the origin, 100 initial conditions
range = 5;
pointsPerDim = 10;
X = linspace(-range,range,pointsPerDim);
[XX,YY] = meshgrid(X,X);
x0 = [XX(:) YY(:)].';

% Number of trajectories
numTraj = size(x0,2);

% Sample time
deltaT = 0.05;

% Time horizon for Reconstruction
Ti = 0:0.1:20; % indirect
Td = 0:0.1:1.5;% direct

% Length of trajectories, random or 5
% tFinal = 1+20*rand(1,numTraj);
tFinal = 5*ones(1,numTraj);

% Code to handle variable length trajectories, appends NaNs to the sample
% time vectors corresponding to shorter trajectories
maxLength = length(0:deltaT:max(tFinal));
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(deltaT,x)).',tFinal,'UniformOutput',false),...
    'UniformOutput', false));

%% MCMC parameters
% Noise standard deviation
standardDeviation = 0.01; 
if standardDeviation == 0
    numTrials = 1;
else
    numTrials = 500; % Number of Monte-Carlo trials
end
% Matrices to store results of Monte-Carlo trials
YRd1 = zeros(stateDimension,length(Td),numTrials); % Direct reconstruction
YRi1 = zeros(stateDimension,length(Ti),numTrials); % Indirect reconstruction
YRd2 = zeros(n,length(Td),numTrials); % Direct reconstruction
YRi2 = zeros(n,length(Ti),numTrials); % Indirect reconstruction
Y = zeros(stateDimension,length(Ti),numTrials); % True trajectory
X = zeros(stateDimension,numTrials); % Initial conditions
RMSd1 = zeros(numTrials,1); %
RMSi1 = zeros(numTrials,1); %
RMSd2 = zeros(numTrials,1); %
RMSi2 = zeros(numTrials,1); %

%% Kernels (see the class Kernel for details)
%%%%%%%%%%%%%%%(R2020b or newer version of MATLAB needed)%%%%%%%%%%%%%%%%%
% Gaussian, second order
% mu = 3;
% Regularization=0.001;
% K = KernelRKHS('Gaussian',mu);

% Exponential, second order
mu2 = 25;
Regularization2 = 7e-6;
K2 = KernelRKHS('Exponential',mu2);

% Exponential, first order
mu1 = 200; 
K1 = KernelRKHS('Exponential',mu1);
Regularization1=1e-8;

%% MCMC Trials
for j = 1:numTrials
    % Generate dataset for DMD 
    %%% type 'help SecondOrderLiouvilleDMD' to see required dataset format %%%
    % Output and derivatives at initial time, for second order DMD
    Output=zeros(1,size(SampleTime,1),numTraj); 
    Derivatives=zeros(1,numTraj);
    % Full state for first order DMD
    State=zeros(2,size(SampleTime,1),numTraj);
    % Generate data
    for i=1:numTraj
        T_i = SampleTime(~isnan(SampleTime(:,i)),i);
        [~,y]=ode45(@(t,x) xDot(t,x),T_i,x0(:,i));
        Output(:,~isnan(SampleTime(:,i)),i)=y(:,1).'+ standardDeviation*randn(size(y(:,1).'));
        State(:,~isnan(SampleTime(:,i)),i)=y.'+ standardDeviation*randn(size(y.'));
        Derivatives(:,i) = y(1,2) + standardDeviation*randn;
    end
    
    %% Liouville DMD
    % Second order
    [~,~,~,r2,f2] = SecondOrderLiouvilleDMD(K2,Output,SampleTime,...
        Derivatives,Regularization2);
    
    %%%%%%%%%%%%%%%%% First order DMD for comparison %%%%%%%%%%%%%%%%%%%%%%
    [~,~,~,r1,f1] = LiouvilleDMD(K1,State,SampleTime,1,Regularization1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Reconstruction
    if standardDeviation ~= 0
        % Random initial condition in [-2,2] x [-2,2]
        x = -2+4*rand(stateDimension,1);
    else
        x = [-1;1];
    end
    
    % Actual trajectory
    [~,y]=ode45(@(t,x) xDot(t,x),Ti,x);
    
    % Direct Reconstruction
    yrd2 = zeros(numel(Td),1);
    yrd1 = zeros(numel(Td),size(y,2));
    for i=1:numel(Td)
        yrd2(i,:) = r2(Td(i),x(1),x(2));
        yrd1(i,:) = r1(Td(i),x);
    end
    
    % Indirect reconstruction first order
    [~,yri1]=ode45(@(t,x) f1(x),Ti,x);
    
    % Indirect reconstruction second order
    f22 = @(x) [x(2);f2(x(1))];
    [~,yri2]=ode45(@(t,x) f22(x),Ti,x);
    
    % Store results for jth trial
    X(:,j)=x; % Store initial condition
    Y(:,:,j) = y.'; % Store true trajectory
    
    % Store reconstructed state trajectories
    YRd1(:,:,j) = yrd1.';        % First order, direct
    YRi1(:,:,j) = yri1.';        % First order, indirect
    YRd2(:,:,j) = yrd2.';        % Second order, direct
    YRi2(:,:,j) = yri2(:,1:n).'; % Second order, indirect
    
    % Store relative RMS output prediction errors
    RMSd1(j) = rms((yrd1(:,1:n)-y(1:numel(Td),1:n))./max(abs(y(:,1:n)))); % First order, direct
    RMSi1(j) = rms((yri1(:,1:n)-y(:,1:n))./max(abs(y(:,1:n)))); % First order, indirect
    RMSd2(j) = rms((yrd2-y(1:numel(Td),1:n))./max(abs(y(:,1:n)))); % Second order, direct
    RMSi2(j) = rms((yri2(:,1:n)-y(:,1:n))./max(abs(y(:,1:n))));% Second order, indirect
end
%% Plots
plot(Ti,y(1:numel(Ti),1),Ti,yri2(1:numel(Ti),1));
legend('True','Reconstructed');
title('Second order');
figure
plot(Ti,y(1:numel(Ti),:),Ti,yri1);
legend('True x1','True x2','Reconstructed x1','Reconstructed x2');
title('First order');

figure
plot(Ti,y(:,1),'LineWidth',2);
hold on
plot(Ti,yri1(:,1),'k-.','LineWidth',2)
plot(Ti,yri2(:,1),'r--','LineWidth',2)
hold off
legend('True','First order DMD','Second order DMD');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
set(gca,'FontSize',12,'TickLabelInterpreter','latex');
filename = ['comparisonNoise' num2str(standardDeviation) 'Duffing.pdf'];
saveas(gcf,filename);

if numTrials > 1
    figure
    boxplot([RMSi1 RMSi2],'Labels',{'First order DMD' 'Second order DMD'})
    ylabel('Relative RMS error','interpreter','latex')
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [6 5]);
    set(gcf, 'PaperPosition', [0 0 6 5]);
    set(gca,'FontSize',12,'TickLabelInterpreter','latex');
    filename = ['comparisonNoise' num2str(standardDeviation) 'DuffingBox.pdf'];
    saveas(gcf,filename);
end

figure
XX = -10:0.5:10;
F2 = zeros(size(XX));
F = zeros(size(XX));
for i = 1:numel(XX)
    F2(i)=f2(XX(i));
    temp = xDot(0,[XX(i);0]);
    F(i) = temp(2);
end
plot(XX,F,XX,F2,'LineWidth',1.5);
xlabel('$x$','interpreter','latex');
l=legend('True ($f(x)$)','Data-driven ($\hat{f}(x)$)');
set(l,'Interpreter','latex');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
set(gca,'FontSize',12,'TickLabelInterpreter','latex');
filename = ['secondOrderLiouvilleDuffingNoise' num2str(standardDeviation) 'VectorField.pdf'];
saveas(gcf,filename);

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end