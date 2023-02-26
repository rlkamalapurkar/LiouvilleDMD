% This script generates trajectories of the Duffing oscillator and uses
% first order and second order Liouville DMD to generate a predictive model 
% from the data.
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
close all
clear all
addpath('../../lib')
%% Generate and format data for DMD

% Duffing oscillator parameters and model
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

numTraj = size(x0,2);
% Sample time
deltaT = 0.05;

% Length of trajectories, random or 5
% tFinal = 1+20*rand(1,numTraj);
tFinal = 5*ones(1,numTraj);

% Code to handle variable length trajectories, appends NaNs to the sample
% time vectors corresponding to shorter trajectories
maxLength = length(0:deltaT:max(tFinal));
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(deltaT,x)).',tFinal,'UniformOutput',false),...
    'UniformOutput', false));

% Initialize data storage
% Output and derivatives at initial time, for second order DMD
Output=zeros(1,size(SampleTime,1),numTraj); 
Derivatives=zeros(1,numTraj);
% Full state for first order DMD
State=zeros(2,size(SampleTime,1),numTraj);

% Generate data
for i=1:numTraj
    T_i = SampleTime(~isnan(SampleTime(:,i)),i);
    [~,y]=ode45(@(t,x) xDot(t,x),T_i,x0(:,i));
    Output(:,~isnan(SampleTime(:,i)),i)=y(:,1).';
    State(:,~isnan(SampleTime(:,i)),i)=y.';
    Derivatives(:,i) = y(1,2);
end

%% Kernels (see the class Kernel for details)
% (R2020b or newer version of MATLAB needed)
% Gaussian, second order
% mu = 3;
% Regularization=0.001;
% K = KernelRKHS('Gaussian',mu);

% Exponential, second order
mu2 = 25;
Regularization2 = 0.000007;
K2 = KernelRKHS('Exponential',mu2);

% Exponential, first order
mu1 = 200; 
K1 = KernelRKHS('Exponential',mu1);
Regularization1=1e-8;

%% Liouville DMD
% Second order
[~,~,~,r2,f2] = SecondOrderLiouvilleDMD(K2,Output,SampleTime,...
    Derivatives,Regularization2);

% First order
[~,~,~,r1,f1] = LiouvilleDMD(K1,State,SampleTime,1,Regularization1);

%% Reconstruction
T = 0:0.1:20;
x = [1;1];
% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),T,x);

% Direct Reconstruction
TDirect = 0:0.1:1.5;
yr2 = zeros(numel(TDirect),1);
yr1 = zeros(numel(TDirect),size(y,2));
for i=1:numel(TDirect)
    yr2(i,:) = real(r2(TDirect(i),x(1),x(2)));
    yr1(i,:) = real(r1(TDirect(i),x));
end

plot(TDirect,y(1:numel(TDirect),1),TDirect,yr2);
legend('True','Reconstructed');
title('Second order');
figure
plot(TDirect,y(1:numel(TDirect),:),TDirect,yr1);
legend('True x1','True x2','Reconstructed x1','Reconstructed x2');
title('First order');

% Indirect reconstruction first order
[~,yri1]=ode45(@(t,x) f1(x),T,x);

% Indirect reconstruction second order
f22 = @(x) [x(2);f2(x(1))];
[~,yri2]=ode45(@(t,x) f22(x),T,x);

figure
plot(T,y(:,1),T,yri2(:,1));
legend('True','Reconstructed');
title('Second order, indirect');
figure
plot(T,y,T,yri1);
legend('True x1','True x2','Reconstructed x1','Reconstructed x2');
title('First order, indirect');

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end