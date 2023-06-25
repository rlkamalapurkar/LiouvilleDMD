% This script generates trajectories of a controlled Duffing oscillator
% uses control Liouville DMD to generate a predictive model for a given 
% feedback controller.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
function controlledDuffing()

addpath('../../lib');

%% Generate Trajectories
n = 2; % Number of dimensions that f maps from/to
m = 1; % Dimensions of the controller
f = @(x) [x(2) ; x(1) - x(1)^3];
g = @(x) [0 ; 2 + sin(x(1))];
IV_selection = 'grid'; 
samp_min = -3;
samp_max = 3;
if strcmp(IV_selection,'grid')
    pointsPerDim = 15;
    XDim = linspace(samp_min,samp_max,pointsPerDim);
    [XX,YY] = meshgrid(XDim,XDim);
    IV = [XX(:) YY(:)].';
    M = size(IV,2);
elseif strcmp(IV_selection,'random')
    % Get TotalTrajectories random IV's.
    M = 100;
    IV = samp_min + (samp_max - samp_min)*rand(n, M);
elseif strcmp(IV_selection,'halton')
    M = 100;
    % Get TotalTrajectories halton sequence
    haltonseq = @(n,d) net(haltonset(d),n);
    halton = haltonseq(M, n);
    IV = samp_min + (samp_max - samp_min)*halton.';
else
    error('Unknown IV selection mode %s', IV_selection)
end
ts = 0.05;
T = 1*ones(1,M);
maxLength = length(0:ts:max(T));
X = zeros(n,maxLength,M);
U = zeros(m,maxLength,M);
for i = 1:M
    freq = 1 + 2*rand(15,1);
    coeff = -1 + 2*rand(15,1);
    phase = -1 + 2*rand(15,1);
    u = @(t) sum(coeff.*sin(t.*freq + phase)); % t is a row vector
    F = @(t,x) f(x) + g(x) * u(t); % The update function
    [t,y] = ode45(F,0:ts:T,IV(:,i));
    X(:,:,i) = y.';
    U(:,:,i) = u(t.');
end
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(ts,x)).',T,'UniformOutput',false), 'UniformOutput', false));
%% Kernels
kT = 10;
e = 1e-8;

K=KernelvvRKHS('Exponential',kT*ones(m+1,1));
KT=KernelRKHS('Exponential',kT);

%% Feedback controller
mu = @(x) -2*x(1,:,:) - 1*x(2,:,:);

%% CLDMD
[~,~,~,r,fHat] = ControlLiouvilleDMD(KT,K,X,U,SampleTime,mu,e);

%% Indirect reconstruction
x0 = [2;-2];
t_pred = 0:0.1:15;
[~,y_pred] = ode45(@(t,x) fHat(x),t_pred,x0);
[~,y] = ode45(@(t,x) f(x) + g(x) * mu(x),t_pred,x0);
y_pred_dir = zeros(size(y));
for i=1:numel(t_pred)
    y_pred_dir(i,:) = r(t_pred(i),x0).';
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

% figure
% plot(t_pred,y-y_pred,'linewidth',2)
% xlabel('Time (s)')
% set(gca,'fontsize',16)
% legend('$x_1(t)-\hat{x}_1(t)$','$x_2(t)-\hat{x}_2(t)$',...
% 'interpreter','latex','fontsize',16,'location','east')

% Data storage
% temp=[t_pred.' y y_pred];
% save('DuffingCLDMDReconstruction.dat','temp','-ascii');
% temp=[t_pred.' y-y_pred];
% save('DuffingCLDMDError.dat','temp','-ascii');

% temp=[t_pred.' y y_pred_dir];
% save('DuffingCLDMDReconstructionDirect.dat','temp','-ascii');
% temp=[t_pred.' (y-y_pred_dir)];
% save('DuffingCLDMDErrorDirect.dat','temp','-ascii');
%% Vector field
% figure
% XDimeval = linspace(-2,2,9);
% [XX,YY] = meshgrid(XDimeval,XDimeval);
% IVeval = [XX(:) YY(:)].';
% x_dot_hat_at_x0 = [];
% x_dot_at_x0 = [];
% for i=1:size(IVeval,2)
%     x0=IVeval(:,i);
%     x_dot_hat_at_x0 = [x_dot_hat_at_x0 fHat(x0)];
%     x_dot_at_x0 = [x_dot_at_x0, f(x0)+g(x0)*mu(x0)];
% end
% max(max(abs(x_dot_at_x0 - x_dot_hat_at_x0)))
% surf(XX,YY,reshape(x_dot_hat_at_x0(2,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(\hat{f}(x) + \hat{g}(x)\mu(x)\right)_2$','interpreter','latex','fontsize',16)
% %f_saveplot('DuffingCLDMD_dim_hat_2')
% set(gca,'fontsize',16)
% figure
% surf(XX,YY,reshape(x_dot_at_x0(2,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(f(x) + g(x)\mu(x)\right)_2$','interpreter','latex','fontsize',16)
% %f_saveplot('DuffingCLDMD_dim_2')
% set(gca,'fontsize',16)
% figure
% surf(XX,YY,reshape(x_dot_hat_at_x0(1,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(\hat{f}(x) + \hat{g}(x)\mu(x)\right)_1$','interpreter','latex','fontsize',16)
% %f_saveplot('DuffingCLDMD_dim_hat_1')
% set(gca,'fontsize',16)
% figure
% surf(XX,YY,reshape(x_dot_at_x0(1,:),9,9))
% xlabel('$x_1$','interpreter','latex','fontsize',16)
% ylabel('$x_2$','interpreter','latex','fontsize',16)
% zlabel('$\left(f(x)+ g(x)\mu(x)\right)_1$','interpreter','latex','fontsize',16)
% %f_saveplot('DuffingCLDMD_dim_1')
% set(gca,'fontsize',16)
% 
% temp = [IVeval.' x_dot_hat_at_x0(2,:).'];
% save('DuffingCLDMDVectorFieldDim2Hat.dat','temp','-ascii');
% temp = [IVeval.' x_dot_at_x0(2,:).'];
% save('DuffingCLDMDVectorFieldDim2.dat','temp','-ascii');
% temp = [IVeval.' (abs(x_dot_at_x0(2,:) -  x_dot_hat_at_x0(2,:))./max(abs(x_dot_at_x0(2,:)))).'];
% save('DuffingCLDMDVectorFieldDim2Error.dat','temp','-ascii');
% temp = [IVeval.' x_dot_hat_at_x0(1,:).'];
% save('DuffingCLDMDVectorFieldDim1Hat.dat','temp','-ascii');
% temp = [IVeval.' x_dot_at_x0(1,:).'];
% save('DuffingCLDMDVectorFieldDim1.dat','temp','-ascii');
% temp = [IVeval.' (abs(x_dot_at_x0(1,:) -  x_dot_hat_at_x0(1,:))./max(abs(x_dot_at_x0(1,:)))).'];
% save('DuffingCLDMDVectorFieldDim1Error.dat','temp','-ascii');
% temp = [IVeval.' vecnorm((x_dot_at_x0 - x_dot_hat_at_x0)./max(vecnorm(x_dot_at_x0))).'];
% save('DuffingCLDMDVectorFieldError.dat','temp','-ascii');
end

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end