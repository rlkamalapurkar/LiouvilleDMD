% This script generates trajectories of the Duffing oscillator and uses
% first order Koopman DMD to generate a predictive model from the data.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function LorenzLiouvilleDMD()
    rng(1)
    close all
addpath('../../lib')

% % Lorenz trajectory generator
sigma = 10;
beta = 8/3;
rho = 28;
xDot = @(t,x) [sigma*(x(2)-x(1)); x(1)*(rho-x(3))-x(2); x(1)*x(2)-beta*x(3)];
n = 3;

range = 30;
pointsPerDim = 21;
numTraj = 400;
IV_selection = 'random'; 
if strcmp(IV_selection,'random')
    % Get TotalTrajectories random IV's.
    x0 = -range + 2*range*rand(n, numTraj);
elseif strcmp(IV_selection,'halton')
    % Get TotalTrajectories halton sequence
    haltonseq = @(n,d) net(haltonset(d),n);
    halton = haltonseq(numTraj, n);
    x0 = -range + 2*range*halton.';
elseif strcmp(IV_selection,'grid')
    x = linspace(-range,range,pointsPerDim);
    [XX,YY,ZZ] = ndgrid(x);
    x0 = [XX(:) YY(:) ZZ(:)].';
    numTraj = size(x0,2);
else
    error('Unknown IV selection mode %s', IV_selection)
end

deltaT = 0.05;
tFinal = 5*ones(1,numTraj);

% Code to handle variable length trajectories, appends NaNs to the sample
% time vectors corresponding to shorter trajectories
maxLength = length(0:deltaT:max(tFinal));
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(deltaT,x)).',tFinal,'UniformOutput',false), 'UniformOutput', false));

% Initialize data storage
State=zeros(n,size(SampleTime,1),numTraj);

% Generate data
for i=1:numTraj
    T_i = SampleTime(~isnan(SampleTime(:,i)),i);
    [~,y]=ode45(@(t,x) xDot(t,x),T_i,x0(:,i));
    State(:,~isnan(SampleTime(:,i)),i)=y.';
end

% % Kernel
mu = 259; 
Kr = KernelRKHS('Exponential',mu);
mu = 260; 
Kd = KernelRKHS('Exponential',mu);
PinvTol = 0;

% mu = 500.0; 
% Kr = KernelRKHS('Gaussian',mu);
Regularization = 1e-8; 

%% Liouville DMD
ScalingFactor = 1;
[Z,S,lsf,rsf,fc] = ConvergentLiouvilleDMD(Kd,Kr,State,SampleTime,PinvTol=PinvTol);

%% Reconstruction
T = 0:0.01:20;
x = [1;1;1];
options = odeset('RelTol',1e-5);
% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),T,x);
% DMD reconstruction
[~,yri1]=ode45(@(t,x) fc(x),T,x,options);

% figure
plot(T,y,"LineStyle","-.","LineWidth",2);
hold on
plot(T,yri1,"LineStyle","-","LineWidth",2);
legend('True x1','True x2','True x3','Reconstructed x1','Reconstructed x2','Reconstructed x3');title('Reconstruction');

end

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end