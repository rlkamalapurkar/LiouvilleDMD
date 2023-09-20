% This script generates trajectories of a two link robot and
% uses control Liouville DMD to generate a predictive model for a given 
% feedback controller.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function [errSVDPinv,errSVDReg,errEig] = ConvergentTwoLinkRobotMCMC(numTrials)
    % rng(1) % for reproducibility
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
    samp_min = -2;
    samp_max = 2;
    M = 200;
    ts = 0.1;
    T = 5*ones(1,M);
    maxLength = length(0:ts:max(T));
    X = zeros(n,maxLength,M);
    U = zeros(m,maxLength,M);
    errSVDReg = zeros(numTrials,1);
    errSVDPinv = zeros(numTrials,1);
    errEig = zeros(numTrials,1);
    
    %% Kernels
    % Best kernel parameters for regularization
    krReg = 5;
    kReg = 10;
    kdReg = 15;
    eReg = 1e-3;
    
    KReg=KernelvvRKHS('Exponential',kReg*ones(m+1,1));
    KrReg=KernelRKHS('Exponential',krReg);
    KdReg=KernelRKHS('Exponential',kdReg);
    
    % Best kernel parameters for pseudoinverse
    krPinv = 2;
    kPinv = 3*krPinv;
    kdPinv = 4*krPinv;
    
    KPinv=KernelvvRKHS('Exponential',kPinv*ones(m+1,1));
    KrPinv=KernelRKHS('Exponential',krPinv);
    KdPinv=KernelRKHS('Exponential',kdPinv);
    
    % Indirect CLDMD for comparison
    kEig = 10;
    eEig = 1e-3;
    KEig=KernelvvRKHS('Exponential',kEig*ones(m+1,1));
    KTEig=KernelRKHS('Exponential',kEig);
    
    %% MCMC trials
    for j=1:numTrials
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
        for i = 1:M
            freq = 1 + 2*rand(30,1);
            coeff = -1 + 2*rand(30,1);
            phase = -1 + 2*rand(30,1);
            u = @(t) [sum(coeff(1:15,:).*sin(t.*freq(1:15,:) + phase(1:15,:)))
                      sum(coeff(16:30,:).*sin(t.*freq(16:30,:) + phase(16:30,:)))];
            F = @(t,x) f(x) + g(x) * u(t); % The update function
            [t,y] = ode45(F,0:ts:T,IV(:,i));
            X(:,:,i) = y.';
            U(:,:,i) = u(t.');
        end
        SampleTime = cell2mat(cellfun(@(x) [x;NaN(maxLength-length(x),1)],...
            arrayfun(@(x) (oddLength(ts,x)).',T,'UniformOutput',false), 'UniformOutput', false));
        
        %% Feedback controller
        c1 = 1 + 9.*rand(1);
        c2 = 1 + 9.*rand(1);
        c3 = 11 + 9.*rand(1);
        c4 = 11 + 9.*rand(1);
        mu = @(x) cat(1, -c1*x(1,:,:) - c2*x(2,:,:), -c3*x(1,:,:) - c4*x(2,:,:));
        
        %% SCLDMD
        [~,~,~,~,fHatSVDPinv] = ConvergentControlLiouvilleDMD(KdPinv,KrPinv,KPinv,X,U,SampleTime,mu);
        [~,~,~,~,fHatSVDReg] = ConvergentControlLiouvilleDMD(KdReg,KrReg,KReg,X,U,SampleTime,mu,RegTol=eReg);
        % Indirect CLDMD for comparison
        [~,~,~,~,fHatEig] = ControlLiouvilleDMD(KTEig,KEig,X,U,SampleTime,mu,eEig);
        
        %% Indirect reconstruction
        x0 = [1;-1;1;-1];
        t_pred = 0:0.05:15;
        [~,y] = ode45(@(t,x) f(x) + g(x) * mu(x),t_pred,x0);
        temp = parfeval(@ode45,2,@(t,x)fHatSVDPinv(x),t_pred,x0);
        % Block for up to maxTime seconds waiting for a result
        maxTime = 0.1;
        didFinish = wait(temp, 'finished', maxTime);
        if ~didFinish
            cancel(temp);
            errSVDPinv(j) = NaN;
        else
            [~,yPredSVDPinv] = fetchOutputs(temp);
            errSVDPinv(j) = rms(vecnorm(y.' - yPredSVDPinv.')/(max(vecnorm(y.'))));
        end

        temp = parfeval(@ode45,2,@(t,x)fHatSVDReg(x),t_pred,x0);
        % Block for up to maxTime seconds waiting for a result
        maxTime = 0.1;
        didFinish = wait(temp, 'finished', maxTime);
        if ~didFinish
            cancel(temp);
            errSVDReg(j) = NaN;
        else
            [~,yPredSVDReg] = fetchOutputs(temp);
            errSVDReg(j) = rms(vecnorm(y.' - yPredSVDReg.')/(max(vecnorm(y.'))));
        end

        temp = parfeval(@ode45,2,@(t,x)fHatEig(x),t_pred,x0);
        % Block for up to maxTime seconds waiting for a result
        maxTime = 0.1;
        didFinish = wait(temp, 'finished', maxTime);
        if ~didFinish
            cancel(temp);
            errEig(j) = NaN;
        else
            [~,yPredEig] = fetchOutputs(temp);
            errEig(j) = rms(vecnorm(y.' - yPredEig.')/(max(vecnorm(y.'))));
        end
        % [~,yPredSVDPinv] = ode45(@(t,x) fHatSVDPinv(x),t_pred,x0);
        % [~,yPredSVDReg] = ode45(@(t,x) fHatSVDReg(x),t_pred,x0);
        % [~,yPredEig] = ode45(@(t,x) fHatEig(x),t_pred,x0);
        
        % errSVDPinv(j) = rms(vecnorm(y.' - yPredSVDPinv.')/(max(vecnorm(y.'))))
        % errSVDReg(j) = rms(vecnorm(y.' - yPredSVDReg.')/(max(vecnorm(y.'))))
        % errEig(j) = rms(vecnorm(y.' - yPredEig.')/(max(vecnorm(y.'))))
    end
end

%% auxiliary functions
function out = oddLength(dt,tf)
    out = 0:dt:tf;
    if mod(numel(out),2)==0
        out = out(1:end-1);
    end
end