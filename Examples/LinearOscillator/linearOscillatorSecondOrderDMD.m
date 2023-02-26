% This script generates trajectories of a linear oscillator and uses
% first and second order Liouville DMD to generate a predictive model from 
% the data.
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
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
standardDeviation = 0.01; % Noise standard deviation
numTraj = size(x0,2); % Number of trajectories
deltaT = 0.5; % Sampling interval
tFinal = 5*ones(1,numTraj); % Time horizon
% tFinal = 1+5*rand(1,numTraj); % Uncomment to use different horizons
maxLength = length(0:deltaT:max(tFinal)); % Length of longest trajectory

% Matrix of sample times as required by the 'secondOrderLiouvilleDMD' function
SampleTime = repmat((0:deltaT:tFinal(1)).',1,numTraj); 

xDot = @(t,x) [x(2);-2*x(1)]; % System dynamics

T = 0:0.01:10; % Time horizon for Reconstruction

numTrials = 500; % Number of Monte-Carlo trials
% Matrices to store results of Monte-Carlo trials
YR_direct = zeros(n,length(T),numTrials); % Direct reconstruction
YR_indirect = zeros(stateDimension,length(T),numTrials); % Indirect reconstruction
Y = zeros(stateDimension,length(T),numTrials); % True trajectory
X = zeros(stateDimension,numTrials); % Initial conditions
RMS_direct = zeros(numTrials,1); %
RMS_indirect = zeros(numTrials,1); %

%% Kernel
%%%%%%%%%%%%%%%(R2020b or newer version of MATLAB needed)%%%%%%%%%%%%%%%%%
mu = 1000;
K = KernelRKHS('Gaussian',mu);
Regularization=1e-6; % Exponential: 0.0001;

%% DMD Monte Carlo trials
for j=1:numTrials
% Create dataset for DMD 
%%% type 'help SecondOrderLiouvilleDMD' to see required dataset format %%%
Trajectories=zeros(n,size(SampleTime,1),numTraj);
Derivatives=zeros(n,numTraj);
State=zeros(stateDimension,size(SampleTime,1),numTraj);

for i=1:numTraj
    T_i = SampleTime(~isnan(SampleTime(:,i)),i);
    [~,y]=ode45(@(t,x) xDot(t,x), T_i, x0(:,i));
    Trajectories(:,~isnan(SampleTime(:,i)),i) = y(:,1).' + standardDeviation*randn(size(y(:,1).'));
    State(:,~isnan(SampleTime(:,i)),i) = y.';
    Derivatives(:,i) = y(1,2) + standardDeviation*randn;
end

%% Liouville DMD
[~,~,~,g,f] = SecondOrderLiouvilleDMD(K,Trajectories,SampleTime,Derivatives,Regularization);

%%%%%%%%%%%%%%%%% First order DMD for comparison %%%%%%%%%%%%%%%%%%%%%%%%%
% mu = 10;
% K1 = @(X,Y) exp(-1/mu*(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - ...
%    2*pagemtimes(X,'transpose',Y,'none')));
% [~,~,~,ReconstructionFunction] = LiouvilleDMD(K1,State,SampleTime);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Reconstruction
% Actual trajectory
x = -1+2*rand(stateDimension,1); % A random initial condition in the unit hypercube
[~,y]=ode45(@(t,x) xDot(t,x), T, x); % True trajectory
yr_direct = zeros(size(y(:,1:n))); % Array to store reconstructed trajectory
%yrFirstOrder = zeros(size(y)); % For first order DMD
% Direct reconstruction
for i=1:numel(T)
    yr_direct(i,:) = real(g(T(i), x(1), x(2)));
    %yrFirstOrder(i,:) = real(ReconstructionFunction(T(i),x)); % for first order DMD
end
% Indirect reconstruction
odefun = @(t,x) [x(2);real(f(x(1)))];
[~,yr_indirect]=ode45(@(t,x) odefun(t,x), T, x); % True trajectory

X(:,j)=x; % Store initial condition
Y(:,:,j) = y.'; % Store true trajectory
YR_direct(:,:,j) = yr_direct; % Store reconstructed trajectory
YR_indirect(:,:,j) = yr_indirect.'; % Store true trajectory
RMS_direct(j) = rms((yr_direct-y(:,1:n))./max(abs(y(:,1:n)))); % Store relative RMS error
RMS_indirect(j) = rms((yr_indirect(:,1:n)-y(:,1:n))./max(abs(y(:,1:n)))); % Store relative RMS error
end

%% Plots

figure
boxplot([RMS_direct RMS_indirect],'Labels',{'Direct Reconstruction' 'Indirect Reconstruction'})
ylabel('Relative RMS error','interpreter','latex')
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
set(gca,'FontSize',12,'TickLabelInterpreter','latex');
filename = 'secondOrderLiouvilleLinearOscillatorRMSBox.pdf';
%saveas(gcf,filename);

figure
plot(T,y(:,1:n).',T,yr_direct.',T,yr_indirect(:,1:n).','LineWidth',1.5)
l=legend('True ($\gamma(t)$)','Direct ($\hat\gamma_D(t)$)','Indirect ($\hat\gamma_I(t)$)');
set(l,'Interpreter','latex');
xlabel('t [s]','Interpreter','latex');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
set(gca,'FontSize',12,'TickLabelInterpreter','latex');
filename = 'secondOrderLiouvilleLinearOscillatorReconstruction.pdf';
%saveas(gcf,filename);

figure
XX = -1.5:0.1:1.5;
FF = zeros(size(XX));
for i = 1:numel(XX)
    FF(i)=f(XX(i));
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
filename = 'secondOrderLiouvilleLinearOscillatorVectorField.pdf';
%saveas(gcf,filename);

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