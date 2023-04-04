% This script generates trajectories of the Duffing oscillator and uses
% first order Koopman DMD to generate a predictive model from the data.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function DuffingOscillatorKoopmanDMD()

addpath('../../lib')

%% Duffing oscillator trajectory generator
alpha = 1;
beta = -1;
delta = 0;
range = 1;
pointsPerDim = 4;%10;
noiseStandardDeviation = 0;
X = linspace(-range,range,pointsPerDim);
[XX,YY] = meshgrid(X,X);
x0 = [XX(:) YY(:)].';
numTraj = size(x0,2);
X=[];
Y=[];
deltaT = 0.5;
T = 0:deltaT:10;
for i=1:numTraj
    xDot = @(t,x) [x(2);-delta*x(2)-beta*x(1)-alpha*x(1)^3];
    [t,y]=ode45(@(t,x) xDot(t,x),T,x0(:,i));
    X = [X y(1:end-1,:).'+noiseStandardDeviation*randn(size(y(1:end-1,:).'))];
    Y = [Y y(2:end,:).'+noiseStandardDeviation*randn(size(y(2:end,:).'))];
end

%% Kernel
mu = 20;
l = 1e-6;
K = KernelRKHS('Exponential',mu);

%% Koopman DMD
[~,~,~,cr,fd,fc] = KoopmanDMD(X,Y,K,deltaT,l);

%% Reconstruction 
tfRec = 20; % Final time for reconstruction
x = [1;-0.75]; % Initial condition for reconstruction

% using eigenvalue
Tc=0:0.1:tfRec;

% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),Tc,x);
ye = zeros(size(y));
for i=1:numel(Tc)
    ye(i,:) = real(cr(Tc(i),x));
end
plot(Tc,y,Tc,ye,'LineWidth',2);
legend('True $x_1$','True $x_2$','$\hat{x}_1$ using eigenvalue','$\hat{x}_2$ using eigenvalue','interpreter','latex')   

% using continuous vector field
[~,ycv]=ode45(@(t,x) fc(x),Tc,x);
figure
plot(Tc,y,Tc,ycv,'LineWidth',2);
legend('True $x_1$','True $x_2$','$\hat{x}_1$ continuous vectorfield','$\hat{x}_2$ continuous vectorfield','interpreter','latex')
% using discrete vector field
Tdv = 0:deltaT:tfRec;
% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),Tdv,x);
ydv = zeros(size(y));
ydv(1,:)=x;
for i=2:numel(Tdv)
    ydv(i,:) = fd(1,ydv(i-1,:).');
end
figure
plot(Tdv,y,Tdv,ydv,'LineWidth',2);
legend('True $x_1$','True $x_2$','$\hat{x}_1$ discrete vectorfield','$\hat{x}_2$ discrete vectorfield','interpreter','latex')
end