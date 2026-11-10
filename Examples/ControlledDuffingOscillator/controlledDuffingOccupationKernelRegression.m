% This script generates trajectories of a controlled Duffing oscillator
% and uses control occupation kernel regression to generate a predictive 
% model for a given control signal.
%
% © Rushikesh Kamalapurkar and Moad Abudia, 2025
function controlledDuffingOccupationKernelRegression()
%rng(1) % delete to randomize
addpath('../../lib');

%% Generate Trajectories
n = 2; % Number of dimensions that f maps from/to
m = 1; % Dimensions of the controller
alpha = 1;
beta = -1;
delta = 0;
f = @(x) [x(2) ; -delta*x(2)-beta*x(1)-alpha*x(1)^3];
g = @(x) [0 ; 2 + sin(x(1))];
F = @(x,u) f(x) + g(x) * u; % The update function
samp_min = -3;
samp_max = 3;
pointsPerDim = 9;
XDim = linspace(samp_min,samp_max,pointsPerDim);
[XX,YY] = meshgrid(XDim,XDim);
X0 = [XX(:) YY(:)].';
M = size(X0,2); % Number of trajectories
ts = 0.05;
Nsamp = 11; % Odd number of samples per trajectory for Simpson's rule
t = repmat((0:ts:(Nsamp-1)*ts).',1,M);
X = zeros(n,Nsamp,M);
U = zeros(m,Nsamp,M);
for i = 1:M
    % Random sinusoidal control signal for each trajectory
    a = -2+4*rand; w = 1+4*rand; p = 2*pi*rand;
    ui = @(tt) a*sin(w*tt+p);
    [~,y] = ode45(@(tt,x) F(x,ui(tt)),t(:,i),X0(:,i));
    X(:,:,i) = y.';
    U(:,:,i) = ui(t(:,i).');
end

%% Kernels
k = 20;
l = 1e-10;

K=KernelvvRKHS('Exponential',k*ones(m+1,1));

%% Controller
u = @(t) 1*sin(t);

%% Control occupation kernel regression
fgHat = ControlOccupationKernelRegression(K,X,U,t,PinvTol = l);

%% Indirect reconstruction
x0 = [2;-2];
t_pred = 0:0.1:15;
[~,y] = ode45(@(tt,x) F(x,u(tt)),t_pred,x0);
[~,y_pred] = ode45(@(tt,x) fgHat(x)*[1;u(tt)],t_pred,x0);
y = y.';
y_pred = y_pred.';

% Plots
plot(t_pred,y,'linewidth',2)
hold on
set(gca,'ColorOrderIndex',1)
plot(t_pred,y_pred,'--','linewidth',2)
hold off
xlabel('Time (s)')
set(gca,'fontsize',16)
legend('$x_1(t)$','$x_2(t)$','$\hat{x}_1(t)$','$\hat{x}_2(t)$',...
'interpreter','latex','fontsize',16,'location','southeast')

end
