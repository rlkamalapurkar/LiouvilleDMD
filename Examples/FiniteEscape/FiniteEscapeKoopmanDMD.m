% This script generates trajectories of a system with finite escape 
% and uses Koopman DMD to generate a predictive model from the data.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
close all
clear all

addpath('../../lib');

%% Finite escape trajectory generator
xrange = 1;
numTraj = 15;
x0 = linspace(-xrange,xrange,numTraj).';
deltaT = 0.25;
tFinal = 0.5;
X=[];
Y=[];
T = 0:deltaT:tFinal;
for i=1:numTraj
    xDot = @(t,x) 1+x.^2;
    [t,y] = ode45(@(t,x) xDot(t,x),T,x0(i));
    X = [X y(1:end-1,:).'];
    Y = [Y y(2:end,:).'];
end

%% Kernel
mu = 2.5; % 1.7 for Williams, 2.5 for Koopman
l=1e-5;
K = KernelRKHS('Exponential',mu);

%% Koopman DMD
[~,~,~,cr,dr,fc] = ...
    KoopmanDMD(X,Y,K,deltaT,l);

%% Reconstruction 
tfRec = 0.7; % Final time for reconstruction
x = -0.5; % Initial condition for reconstruction

% using eigenvalue  
Tc=0:0.01:tfRec;
% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),Tc,x);
ye = zeros(size(y));
for i=1:numel(Tc)
    ye(i,:) = cr(Tc(i),x);
end
plot(Tc,y,Tc,ye,'LineWidth',2);
legend('True $x$','$\hat{x}$ using eigenvalue','interpreter','latex')   

% using continuous vector field
[~,ycv]=ode45(@(t,x) fc(x),Tc,x);
figure
plot(Tc,y,Tc,ycv,'LineWidth',2);
legend('True $x$','$\hat{x}$ continuous vectorfield','interpreter','latex') 

% using discrete vector field
Tdv = 0:deltaT:tfRec;
% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),Tdv,x);
ydv = zeros(size(y));
ydv(1,:)=x;
for i=2:numel(Tdv)
    ydv(i,:) = dr(1,ydv(i-1,:).');
end
figure
plot(Tdv,y,Tdv,ydv,'LineWidth',2);
legend('True $x$','$\hat{x}$ discrete vectorfield','interpreter','latex') 