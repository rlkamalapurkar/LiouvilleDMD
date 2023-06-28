% This script generates trajectories of the Duffing oscillator and uses
% Convergent DMD to generate a predictive model from the data.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function DuffingOscillatorConvergentDMDComparison()

addpath('../../lib')
%% Generate and format data for DMD

% Duffing oscillator parameters and model
alpha = 1;
beta = -1;
delta = 0.011;
xDot = @(t,x) [x(2);-delta*x(2)-beta*x(1)-alpha*x(1)^3];

% Initial conditions, 10 grid points per dimension in a 5 x 5 square 
% centered at the origin, 100 initial conditions
range = 5;
pointsPerDim = 10;
X = linspace(-range,range,pointsPerDim);
[XX,YY] = meshgrid(X,X);
x0 = [XX(:) YY(:)].';

numTraj = size(x0,2);
% Sample time
deltaT = 0.05;

% Length of trajectories, random or 10
% tFinal = 1+20*rand(1,numTraj);
tFinal = 10*ones(1,numTraj);

% Code to handle variable length trajectories, appends NaNs to the sample
% time vectors corresponding to shorter trajectories
maxLength = length(0:deltaT:max(tFinal));
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(deltaT,x)).',tFinal,'UniformOutput',false), 'UniformOutput', false));

% Initialize data storage
% First state variable and derivatives at initial time, for second order DMD
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

% Exponential, first order

% mu = 2.0; 
% Kr = KernelRKHS('Gaussian',mu);
% mu = 2.1; 
% Kd = KernelRKHS('Gaussian',mu);
% Regularization = 0;

mur = 309; 
Kr = KernelRKHS('Exponential',mur);
mud = 310; 
Kd = KernelRKHS('Exponential',mud);
Regularization = 1e-8;

% mu = 1;
% degree = 3;
% bias = 1;
% Kr = KernelRKHS('Polynomial',[mu,degree,bias]);
% mu = 2;
% Kd = KernelRKHS('Polynomial',[mu,degree,bias]);
% Regularization = 1e-6;

%% Liouville DMD
ScalingFactor = 1;
tic
[Z,S,lsf,rsf,fc] = ConvergentLiouvilleDMD(Kd,Kr,State,SampleTime,Regularization);
toc
[~,~,~,~,f1] = LiouvilleDMD(Kr,State,SampleTime,ScalingFactor,Regularization);

%% Reconstruction
T = 0:0.1:20;
x = [3;-2];
options = odeset('RelTol',1e-5);
% Actual trajectory
[~,y]=ode45(@(t,x) xDot(t,x),T,x);

% SDMD reconstruction
[~,yric]=ode45(@(t,x) fc(x),T,x,options);
 
% figure
% plot(T,y,T,yric);
% legend('True $x_1$','True $x_2$','Reconstructed $x_1$','Reconstructed $x_2$','Interpreter','latex','Location','northwest');
% xlabel('Time [s]','Interpreter','latex')
% f_saveplot('DuffingConvergentReconstruction','linewidth',1.2,'fontsize',14)
% temp=[T.' y yric];
% save('DuffingConvergentReconstruction.dat','temp','-ascii');

% DMD reconstruction
[~,yri1]=ode45(@(t,x) f1(x),T,x,options);

% figure
% plot(T,y,T,yri1);legend('True x1','True x2','Reconstructed x1','Reconstructed x2');title('Reconstruction');

% Reconstruction errors
figure
plot(T,vecnorm(y.'-yric.'),T,vecnorm(y.'-yri1.'));
legend('Convergent Liouville DMD','Liouville DMD','Interpreter','latex','Location','northwest');
ylabel('Norm of Reconstruction Error','Interpreter','latex');
xlabel('Time [s]','Interpreter','latex')
% f_saveplot('DuffingConvergentErrorComparison','linewidth',1.5,'fontsize',14)
% temp=[T.' vecnorm(y.'-yric.').',vecnorm(y.'-yri1.').'];
% save('DuffingConvergentErrorComparison.dat','temp','-ascii');

%% Vector field
GridSize = 25;
XDimeval = linspace(-range,range,GridSize);
[XX,YY] = meshgrid(XDimeval,XDimeval);
IVeval = [XX(:) YY(:)].';
x_dot_hat_at_x0_c = [];
x_dot_hat_at_x0 = [];
x_dot_at_x0 = [];
for i=1:size(IVeval,2)
    x0=IVeval(:,i);
    x_dot_hat_at_x0_c = [x_dot_hat_at_x0_c fc(x0)];
    x_dot_hat_at_x0 = [x_dot_hat_at_x0 f1(x0)];
    x_dot_at_x0 = [x_dot_at_x0, xDot(0,x0)];
end
temp = [IVeval.' vecnorm((x_dot_at_x0 - x_dot_hat_at_x0_c)./max(vecnorm(x_dot_at_x0))).'];
disp(['Maximum vectorfield estimation error for convergent DMD is ' num2str(max(max(abs(x_dot_at_x0 - x_dot_hat_at_x0_c))))])
disp(['Maximum vectorfield estimation error for DMD is ' num2str(max(max(abs(x_dot_at_x0 - x_dot_hat_at_x0))))])
% figure
% surf(XX,YY,reshape(vecnorm((x_dot_at_x0 - x_dot_hat_at_x0_c)./max(vecnorm(x_dot_at_x0))),GridSize,GridSize),'EdgeColor','none','FaceColor','interp')
% xlabel('$x_1$','interpreter','latex','fontsize',14)
% ylabel('$x_2$','interpreter','latex','fontsize',14)
% zlabel('Relative Error Norm','interpreter','latex','fontsize',14)
% f_saveplot('DuffingConvergentVectorField','fontsize',14)
% save('DuffingConvergentVectorField.dat','temp','-ascii');

end

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end