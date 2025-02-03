% This script generates trajectories of a two link robot and
% uses control Liouville DMD to generate a predictive model for a given 
% control signal.
%
% © Rushikesh Kamalapurkar and Moad Abudia, 2024
%
function OpenLoopConvergentTwoLinkRobot()
rng(1) % Added to reproduce the plots in the paper, comment out for random results
addpath('../../lib')
%% Generate Trajectories
n = 4; % Number of dimensions that f maps from/to
m = 2; % Dimensions of the controller
f = @(x) ...
    [x(3);
     x(4);
    (1/(0.196^2 - 3.473*0.196 + 0.242^2*cos(x(2))^2)) *...
    [-0.196                   0.196 + 0.242*cos(x(2));
      0.196+0.242*cos(x(2))   -3.473-2*0.242*cos(x(2))] *...
    ((-[-0.242*sin(x(2))*x(4)   -0.242*sin(x(2))*(x(3)+x(4));
         0.242*sin(x(2))*x(3)    0                           ] - ...
    diag([5.3,1.1])) *...
    [x(3);
     x(4)] - ...
    [8.45*tanh(x(3));
     2.35*tanh(x(4))])];
g = @(x) ...
    [0 0;
     0 0;
    (1/(0.196^2 - 3.473*0.196 + 0.242^2*cos(x(2))^2))*...
    [-0.196                     0.196 + 0.242*cos(x(2))  ;...
      0.196 + 0.242*cos(x(2))  -3.473 - 2*0.242*cos(x(2))]];
IV_selection = 'halton'; 
samp_min = -1;
samp_max = 1;
M = 200;
if strcmp(IV_selection,'random')
    % Get TotalTrajectories random IV's.
    IV = samp_min + (samp_max - samp_min)*rand(n, M);
elseif strcmp(IV_selection,'halton')
    % Get TotalTrajectories halton sequence
    haltonseq = @(n,d) net(haltonset(d),n);
    halton = haltonseq(M, n);
    IV = samp_min + (samp_max - samp_min)*halton.';
else
    error('Unknown IV selection mode %s', IV_selection)
end
ts = 0.1;
T = 5*ones(1,M);
maxLength = length(0:ts:max(T));
X = zeros(n,maxLength,M);
U = zeros(m,maxLength,M);
for i = 1:M
    freq = 1 + 2*rand(30,1);
    coeff = -1 + 2*rand(30,1);
    phase = -1 + 2*rand(30,1);
    u = @(t) [sum(coeff(1:15,:).*sin(t.*freq(1:15,:) + phase(1:15,:)))
              sum(coeff(16:30,:).*sin(t.*freq(16:30,:) + phase(16:30,:)))];
    F = @(t,x) f(x) + g(x) * u(t); % The update function
    [t,y] = ode45(F,0:ts:T(i),IV(:,i));
    X(:,:,i) = y.';
    U(:,:,i) = u(t.');
end
SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
    arrayfun(@(x) (oddLength(ts,x)).',T,'UniformOutput',false), 'UniformOutput', false));

%% Kernels

% % Best kernel parameters for regularization
% k = 10;
% kd = 15;
% e = 1e-3;

% Best kernel parameters for pseudoinverse
k = 10;
kd = 12;
e = 0;

K=KernelvvRKHS('Exponential',k*ones(m+1,1));
Kd=KernelRKHS('Exponential',kd);

%% SCLDMD
[~,~,~,~,fHat_SVD] = ConvergentControlLiouvilleDMDOpenLoop(Kd,K,X,U,SampleTime,RegTol=e);

%% Indirect reconstruction
x0 = [1;-1;1;-1];
t_pred = 0:0.05:15;
u = @(t) [sin(t); sin(2*t)];
[~,y_pred_SVD] = ode45(@(t,x) fHat_SVD(x)*[1;u(t)],t_pred,x0);
[~,y] = ode45(@(t,x) f(x) + g(x) * u(t),t_pred,x0);

% temp=[t_pred.' y y_pred_SVD];
% save('2LinkSCLDMDReconstruction.dat','temp','-ascii');
% temp=[t_pred.' y-y_pred_SVD];
% save('2LinkSCLDMDError.dat','temp','-ascii');
% % temp=[t_pred.' (y-y_pred_SVD)/max(vecnorm(y.'-y_pred_SVD.'))];
% % save('2LinkSCLDMDNormalizedError.dat','temp','-ascii');
% 
% temp=[t_pred.' y y_pred_Eig];
% save('2LinkCLDMDReconstruction.dat','temp','-ascii');
% temp=[t_pred.' y-y_pred_Eig];
% save('2LinkCLDMDError.dat','temp','-ascii');
% % temp=[t_pred.' (y-y_pred_Eig)/max(vecnorm(y.'-y_pred_Eig.'))];
% % save('2LinkCLDMDNormalizedError.dat','temp','-ascii');
% 
% temp=[t_pred.' y y_pred_Eig_dir];
% save('2LinkCLDMDReconstructionDirect.dat','temp','-ascii');
% temp=[t_pred.' y-y_pred_Eig_dir];
% save('2LinkCLDMDErrorDirect.dat','temp','-ascii');
% % temp=[t_pred.' (y-y_pred_Eig_dir)/max(vecnorm(y.'-y_pred_Eig_dir.'))];
% % save('2LinkCLDMDNormalizedErrorDirect.dat','temp','-ascii');

% Plots
figure
plot(t_pred,y,'linewidth',2)
hold on
set(gca,'ColorOrderIndex',1)
plot(t_pred,y_pred_SVD,'--','linewidth',2)
hold off
xlabel('Time (s)')
set(gca,'fontsize',16)
legend('$x_1(t)$','$x_2(t)$','$x_3(t)$','$x_4(t)$',...
    '$\hat{x}_1(t)$','$\hat{x}_2(t)$','$\hat{x}_3(t)$','$\hat{x}_4(t)$',...
'interpreter','latex','fontsize',16,'location','southeast')

figure
plot(t_pred,y-y_pred_SVD,'linewidth',2)
xlabel('Time (s)')
set(gca,'fontsize',16)
legend('$x_1(t)-\hat{x}_1(t)$','$x_2(t)-\hat{x}_2(t)$',...
    '$x_3(t)-\hat{x}_3(t)$','$x_4(t)-\hat{x}_4(t)$',...
'interpreter','latex','fontsize',16,'location','east')
end

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end