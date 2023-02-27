% This script generates trajectories of a linear oscillator and uses
% first and second order Liouville DMD to generate a predictive model from 
% the data.
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
clear all
close all
addpath('../../lib')

%% Generate trajectories of a linear oscillator
rng(1); % Seed for random number generator, 
% added to ensure this code generates the exact same plot as the paper

n = 1; % Output dimension
order = 2; % Order of the system (code only for order = 2)
stateDimension = order*n; % State space dimension

% Generate initial conditions on a grid (code specific to order = 2)
range = 1; % grid size
pointsPerDim = 4; % grid points per dimension
X = linspace(-range,range,pointsPerDim);
[XX,YY] = meshgrid(X,X);
x0 = [XX(:) YY(:)].'; % Initial conditions

% Parameters of the trajectories dataset
numTraj = size(x0,2); % Number of trajectories
deltaT = 0.5; % Sampling interval
tFinal = 5*ones(1,numTraj); % Time horizon
% tFinal = 1+5*rand(1,numTraj); % Uncomment to use different horizons
maxLength = length(0:deltaT:max(tFinal)); % Length of longest trajectory

% Matrix of sample times as required by the 'secondOrderLiouvilleDMD' function
SampleTime = repmat((0:deltaT:tFinal(1)).',1,numTraj); 

xDot = @(t,x) [x(2);-2*x(1)]; % System dynamics

T = 0:0.01:20; % Time horizon for Reconstruction

% MCMC parameters
% Noise standard deviation
standardDeviation = 0.01; 
if standardDeviation == 0
    numTrials = 1;
else
    numTrials = 500; % Number of Monte-Carlo trials
end
% Matrices to store results of Monte-Carlo trials
YRd1 = zeros(stateDimension,length(T),numTrials); % Direct reconstruction
YRi1 = zeros(stateDimension,length(T),numTrials); % Indirect reconstruction
YRd2 = zeros(n,length(T),numTrials); % Direct reconstruction
YRi2 = zeros(n,length(T),numTrials); % Indirect reconstruction
Y = zeros(stateDimension,length(T),numTrials); % True trajectory
X = zeros(stateDimension,numTrials); % Initial conditions
RMSd1 = zeros(numTrials,1); %
RMSi1 = zeros(numTrials,1); %
RMSd2 = zeros(numTrials,1); %
RMSi2 = zeros(numTrials,1); %

%% Kernels (see the class Kernel for details)
%%%%%%%%%%%%%%%(R2020b or newer version of MATLAB needed)%%%%%%%%%%%%%%%%%
% Second order
mu2 = 100;
K2 = KernelRKHS('Gaussian',mu2);
Regularization2=1e-8; % Exponential: 0.0001;

% First order
mu1 = 100;
K1 = KernelRKHS('Gaussian',mu2);
Regularization1=1e-6; % Exponential: 0.0001;

%% DMD Monte Carlo trials
for j=1:numTrials
% Generate dataset for DMD 
%%% type 'help SecondOrderLiouvilleDMD' to see required dataset format %%%
% Output and derivatives at initial time, for second order DMD
Output=zeros(n,size(SampleTime,1),numTraj);
Derivatives=zeros(n,numTraj);
% Full state for first order DMD
State=zeros(stateDimension,size(SampleTime,1),numTraj);
% Generate data
for i=1:numTraj
    T_i = SampleTime(~isnan(SampleTime(:,i)),i);
    [~,y]=ode45(@(t,x) xDot(t,x), T_i, x0(:,i));
    Output(:,~isnan(SampleTime(:,i)),i) = y(:,1).' + standardDeviation*randn(size(y(:,1).'));
    State(:,~isnan(SampleTime(:,i)),i) = y.' + standardDeviation*randn(size(y.'));
    Derivatives(:,i) = y(1,2) + standardDeviation*randn;
end

%% Liouville DMD
% Second order
[~,~,~,r2,f2] = SecondOrderLiouvilleDMD(K2,Output,SampleTime,Derivatives,Regularization2);

%%%%%%%%%%%%%%%%% First order DMD for comparison %%%%%%%%%%%%%%%%%%%%%%%%%
[~,~,~,r1,f1] = LiouvilleDMD(K1,State,SampleTime,1,Regularization1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Reconstruction

if standardDeviation ~= 0
    % Random initial condition in [-1,1] x [-1,1]
    x = -1+2*rand(stateDimension,1);
else
    x = [-1;1];
end

% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),T,x);

% Direct Reconstruction
yrd2 = zeros(numel(T),1);
yrd1 = zeros(numel(T),size(y,2));
for i=1:numel(T)
    yrd2(i,:) = r2(T(i),x(1),x(2));
    yrd1(i,:) = r1(T(i),x);
end

% Indirect reconstruction first order
[~,yri1]=ode45(@(t,x) f1(x),T,x);

% Indirect reconstruction second order
f22 = @(x) [x(2);f2(x(1))];
[~,yri2]=ode45(@(t,x) f22(x),T,x);

% Store results for jth trial
X(:,j)=x; % Store initial condition
Y(:,:,j) = y.'; % Store true trajectory

% Store reconstructed state trajectories
YRd1(:,:,j) = yrd1.';        % First order, direct
YRi1(:,:,j) = yri1.';        % First order, indirect
YRd2(:,:,j) = yrd2.';        % Second order, direct
YRi2(:,:,j) = yri2(:,1:n).'; % Second order, indirect

% Store relative RMS output prediction errors
RMSd1(j) = rms((yrd1(:,1:n)-y(1:numel(T),1:n))./max(abs(y(:,1:n)))); % First order, direct
RMSi1(j) = rms((yri1(:,1:n)-y(:,1:n))./max(abs(y(:,1:n)))); % First order, indirect
RMSd2(j) = rms((yrd2-y(1:numel(T),1:n))./max(abs(y(:,1:n)))); % Second order, direct
RMSi2(j) = rms((yri2(:,1:n)-y(:,1:n))./max(abs(y(:,1:n))));% Second order, indirect
end

%% Plots
figure
plot(T,y(:,1),'LineWidth',2);
hold on
plot(T,yri1(:,1),'k-.','LineWidth',2)
plot(T,yri2(:,1),'r--','LineWidth',2)
hold off
legend('True','First order DMD','Second order DMD');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
set(gca,'FontSize',12,'TickLabelInterpreter','latex');
filename = ['comparisonNoise' num2str(standardDeviation) 'LinearOscillator.pdf'];
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
    filename = ['comparisonNoise' num2str(standardDeviation) 'LinearOscillatorBox.pdf'];
    saveas(gcf,filename);
end

figure
XX = -3:0.1:3;
FF = zeros(size(XX));
for i = 1:numel(XX)
    FF(i)=f2(XX(i));
end
plot(XX,-2*XX,XX,FF,'LineWidth',1.5);
xlabel('$x$','interpreter','latex');
l=legend('True ($f(x)$)','Data-driven ($\hat{f}(x)$)');
set(l,'Interpreter','latex');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
set(gca,'FontSize',12,'TickLabelInterpreter','latex');
filename = ['secondOrderLiouvilleLinearOscillatorNoise' num2str(standardDeviation) 'VectorField.pdf'];
saveas(gcf,filename);

% plot(RMS_direct,'*')
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperSize', [6 5]);
% set(gcf, 'PaperPosition', [0 0 6 5]);
% set(gca,'FontSize',12,'TickLabelInterpreter','latex');
% xlabel('Trial','Interpreter','latex');
% ylabel('RMS error','Interpreter','latex');
% filename = 'secondOrderLiouvilleLinearOscillatorRMSDirect.pdf';
% saveas(gcf,filename);
% 
% figure
% plot(RMS_indirect,'*')
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperSize', [6 5]);
% set(gcf, 'PaperPosition', [0 0 6 5]);
% set(gca,'FontSize',12,'TickLabelInterpreter','latex');
% xlabel('Trial','Interpreter','latex');
% ylabel('RMS error','Interpreter','latex');
% filename = 'secondOrderLiouvilleLinearOscillatorRMSIndirect.pdf';
% saveas(gcf,filename);