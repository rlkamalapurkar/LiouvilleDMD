% This script generates trajectories of a controlled Duffing oscillator
% uses control Liouville DMD to generate a predictive model for a given 
% open-loop controller.
%
% © Rushikesh Kamalapurkar 
function convergentOpenLoopControlledDuffing()
%rng(1) % added to reproduce the plots in the paper, delete to randomize
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
    [t,y] = ode45(F,0:ts:T(i),IV(:,i));
    X(:,:,i) = y.';
    U(:,:,i) = u(t.');
end
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(ts,x)).',T,'UniformOutput',false), 'UniformOutput', false));
%% Kernels

% Best kernel parameters for pseudoinverse
kd = 11;
Kd=KernelRKHS('Exponential',kd);
k = 10;
K=KernelvvRKHS('Exponential',k*ones(m+1,1));

% Best kernel parameters for regularization
% e = 1e-8;
% kd = 20;
% Kd=KernelRKHS('Exponential',kd);
% k = 15;
% K=KernelvvRKHS('Exponential',k*ones(m+1,1));

%% SCLDMD
[~,~,~,~,fg_SVD] = ConvergentControlLiouvilleDMDOpenLoop(Kd,K,X,U,SampleTime);

%% Indirect reconstruction
x0 = [2;-2];
t_pred = 0:0.1:10;
u = @(t) sin(t) + cos(2*t);
[~,y_pred_SVD] = ode45(@(t,x) fg_SVD(x)*[1;u(t)],t_pred,x0);
[~,y] = ode45(@(t,x) f(x) + g(x)*u(t),t_pred,x0);

%% Plots
plot(t_pred,y,'linewidth',1)
hold on
set(gca,'ColorOrderIndex',1)
plot(t_pred,y_pred_SVD,'--','linewidth',2)
hold off
xlabel('Time (s)')
set(gca,'fontsize',16)
legend('$x_1(t)$','$x_2(t)$','$\hat{x}_1(t)$','$\hat{x}_2(t)$',...
'interpreter','latex','fontsize',16,'location','southeast')
title('SCLDMD Prediction')

figure
plot(t_pred,y-y_pred_SVD,'linewidth',2)
xlabel('Time (s)')
set(gca,'fontsize',16)
legend('$x_1(t)-\hat{x}_1(t)$','$x_2(t)-\hat{x}_2(t)$',...
'interpreter','latex','fontsize',16,'location','east')
title('SCLDMD Prediction Error')

% % Store plot data for LaTeX
% temp=[t_pred.' y y_pred_SVD];
% save('DuffingSCLDMDReconstruction.dat','temp','-ascii');
% temp=[t_pred.' y-y_pred_SVD];
% save('DuffingSCLDMDError.dat','temp','-ascii');

%% Vector field
nEval = 9;
XDimeval = linspace(-2,2,nEval);
[XX,YY] = meshgrid(XDimeval,XDimeval);
IVeval = [XX(:) YY(:)].';
f_hat_at_x0_SVD = [];
g_hat_at_x0_SVD = [];
f_at_x0 = [];
g_at_x0 = [];
for i=1:size(IVeval,2)
    x0=IVeval(:,i);
    temp = fg_SVD(x0);
    f_hat_at_x0_SVD = [f_hat_at_x0_SVD temp(:,1)];
    f_at_x0 = [f_at_x0, f(x0)];
    g_hat_at_x0_SVD = [g_hat_at_x0_SVD temp(:,2)];
    g_at_x0 = [g_at_x0, g(x0)];
end
disp(['SVD f error is ' num2str(max(max(abs(f_at_x0 - f_hat_at_x0_SVD))))])
disp(['SVD g error is ' num2str(max(max(abs(g_at_x0 - g_hat_at_x0_SVD))))])
% 
% % Vector field plots
figure
gError = vecnorm((g_at_x0 - g_hat_at_x0_SVD)./max(vecnorm(g_at_x0)));
surf(XX,YY,reshape(gError,nEval,nEval))
xlabel('$x_1$','interpreter','latex','fontsize',16)
title('Error in the estimation of g')
figure
fError = vecnorm((f_at_x0 - f_hat_at_x0_SVD)./max(vecnorm(f_at_x0)));
surf(XX,YY,reshape(fError,nEval,nEval))
xlabel('$x_1$','interpreter','latex','fontsize',16)
title('Error in the estimation of f')

% % Store plot data for LaTeX
% temp = [IVeval.' fError.'];
% save('DuffingVectorFieldfError.dat','temp','-ascii');
% temp = [IVeval.' gError.'];
% save('DuffingVectorFieldgError.dat','temp','-ascii');
end
%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end