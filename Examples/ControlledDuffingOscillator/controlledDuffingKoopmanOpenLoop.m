% This script generates trajectories of a controlled Duffing oscillator
% and uses control Koopman DMD to generate a predictive model for a given 
% control signal.
%
% © Rushikesh Kamalapurkar and Moad Abudia, 2025
function controlledDuffingKoopmanOpenLoop()
%rng(1) % added to reproduce the plots in the paper, delete to randomize
addpath('../../lib');

%% Generate Trajectories
n = 2; % Number of dimensions that f maps from/to
m = 1; % Dimensions of the controller
alpha = 1;
beta = -1;
delta = 0;
f = @(x) [x(2) ; -delta*x(2)-beta*x(1)-alpha*x(1)^3];
g = @(x) [0 ; 2 + sin(x(1))];
IV_selection = 'grid'; 
samp_min = -3;
samp_max = 3;
if strcmp(IV_selection,'grid')
    pointsPerDim = 9;
    XDim = linspace(samp_min,samp_max,pointsPerDim);
    [XX,YY] = meshgrid(XDim,XDim);
    X = [XX(:) YY(:)].';
    M = size(X,2);
elseif strcmp(IV_selection,'random')
    % Get TotalTrajectories random IV's.
    M = 100;
    X = samp_min + (samp_max - samp_min)*rand(n, M);
elseif strcmp(IV_selection,'halton')
    M = 100;
    % Get TotalTrajectories halton sequence
    haltonseq = @(n,d) net(haltonset(d),n);
    halton = haltonseq(M, n);
    X = samp_min + (samp_max - samp_min)*halton.';
else
    error('Unknown IV selection mode %s', IV_selection)
end
ts = 0.1;
U = -2+4*rand(1,M);
Y = zeros(size(X));
for i = 1:M
    F = @(x,u) f(x) + g(x) * u; % The update function
    [~,y] = ode45(@(t,x) F(x,U(:,i)),[0,ts],X(:,i));
    Y(:,i) = y(end,:).';
end
%% Kernels
kT = 20;
k = 20;
l = 1e-6;

K=KernelvvRKHS('Exponential',k*ones(m+1,1));
KT=KernelRKHS('Exponential',kT);

%% Controller
u = @(t) 1*sin(t);

%% CLDMD
fgHat = ControlKoopmanDMDOpenLoop(KT,K,X,U,Y,PinvTol = l);

%% Indirect reconstruction
x0 = [2;-2];
t_pred = 0:ts:15;
U = u(t_pred);
y = zeros(n,numel(t_pred));
y(:,1) = x0;
y_pred = y;
for i=1:numel(t_pred)-1
    F = @(x,u) f(x) + g(x) * u; % The update function
    [~,temp] = ode45(@(t,x) F(x,U(i)),[0,ts],y(:,i));
    y(:,i+1) = temp(end,:).';
    y_pred(:,i+1) = fgHat(y_pred(:,i))*[1;U(i)];
end

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

%% Vector Field Plot
% XDimeval = linspace(-2,2,9);
% [XX, YY] = meshgrid(XDimeval,XDimeval);
% IVeval = [XX(:) YY(:)].';
% x_dot_hat_at_x0 = [];
% x_dot_at_x0 = [];
% for i=1:size(IVeval,2)
%     x0=IVeval(:,i);
%     temp = fgHat(x0);
%     x_dot_hat_at_x0 = [x_dot_hat_at_x0, temp(:,1)];
%     [~,temp] = ode45(@(t,x) F(x,u(t)),[0,ts],x0);
%     x_dot_at_x0 = [x_dot_at_x0, temp(end,:).'];
% end
% figure
% subplot(2,2,1);
% surf(XX,YY,reshape(x_dot_hat_at_x0(1,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(\hat{f}(x) + \hat{g}(x)\mu(x)\right)_1$','interpreter','latex','fontsize',16)
% set(gca,'fontsize',16)
% subplot(2,2,2);
% surf(XX,YY,reshape(x_dot_at_x0(1,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(f(x)+ g(x)\mu(x)\right)_1$','interpreter','latex','fontsize',16)
% set(gca,'fontsize',16)
% subplot(2,2,3);
% surf(XX,YY,reshape(x_dot_hat_at_x0(2,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(\hat{f}(x) + \hat{g}(x)\mu(x)\right)_2$','interpreter','latex','fontsize',16)
% set(gca,'fontsize',16)
% subplot(2,2,4);
% surf(XX,YY,reshape(x_dot_at_x0(2,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(f(x) + g(x)\mu(x)\right)_2$','interpreter','latex','fontsize',16)
% set(gca,'fontsize',16)

end