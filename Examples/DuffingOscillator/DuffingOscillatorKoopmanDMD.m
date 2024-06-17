% This script generates trajectories of the Duffing oscillator and uses
% first order Koopman DMD to generate a predictive model from the data.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function DuffingOscillatorKoopmanDMD()
    % rng(1) % to reproduce the plots in the paper exactly
    addpath('../../lib')
    % % Duffing oscillator trajectory generator
    alpha = 1;
    beta = -1;
    delta = 0.1;
    range = 1;
    pointsPerDim = 4;
    noiseStandardDeviation = 0.001;
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
        [~,y]=ode45(@(t,x) xDot(t,x),T,x0(:,i));
        X = [X y(1:end-1,:).'+noiseStandardDeviation*randn(size(y(1:end-1,:).'))];
        Y = [Y y(2:end,:).'+noiseStandardDeviation*randn(size(y(2:end,:).'))];
    end
    
    % % Kernel
    mu = 20;
    l = 1e-6;
    K = KernelRKHS('Exponential',mu);
    lW = 1e-3;
    KW = KernelRKHS('Exponential',mu);
    
    % % Koopman DMD
    pinv = 0;
    if pinv
        [~,~,~,~,drK,fcK] = KoopmanDMD(X,Y,K,deltaT,PinvTol=lW);
    else
        [~,~,~,~,drK,fcK] = KoopmanDMD(X,Y,K,deltaT,RegTol=l);
    end
    [~,~,~,~,drW,fcW] = WilliamsKDMD(X,Y,KW,deltaT,lW);
    
    % % Reconstruction 
    tfRec = 10; % Final time for reconstruction
    XRec = [-0.75 + 1.5*rand(1,100);-0.75 + 1.5*rand(1,100)]; % Initial conditions for reconstruction
    % Time vector for reconstruction
    Tc = 0:0.1:tfRec;
    Td = 0:deltaT:tfRec;
    KErr = zeros(100,numel(Tc));
    WErr = zeros(100,numel(Tc));
    KErrd = zeros(100,numel(Td));
    WErrd = zeros(100,numel(Td));
    WKDiff = zeros(100,numel(Tc));
    for i=1:100
        x = XRec(:,i);
        % Actual trajectory
        [~,y]=ode45(@(t,x) xDot(t,x),Tc,x);
        maxNorm = max(vecnorm(y.'));
        % Actual trajectory discrete
        [~,yd]=ode45(@(t,x) xDot(t,x),Td,x);
        maxNormd = max(vecnorm(yd.'));
        % using continuous kernel vector field
        [~,ycK]=ode45(@(t,x) fcK(x),Tc,x);
        KErr(i,:) = vecnorm(y.'-ycK.')/maxNorm;
        % using continuous Williams vector field
        [~,ycW]=ode45(@(t,x) fcW(x),Tc,x);
        WErr(i,:) = vecnorm(y.'-ycW.')./maxNorm;
        WKDiff(i,:) = vecnorm(ycK.'-ycW.')./maxNorm;
        % using discrete kernel vector field
        KErrd(i,1) = 0;
        xK = x;
        for j=2:numel(Td)
            xK = drK(1,xK);
            KErrd(i,j) = norm(yd(j,:).'-xK)/maxNormd;
        end
        % using discrete Williams vector field
        WErrd(i,1) = 0;
        xW = x;
        for j=2:numel(Td)
            xW = drW(1,xW);
            WErrd(i,j) = norm(yd(j,:).'-xW)/maxNormd;
        end
    end
    Tc2 = [Tc, fliplr(Tc)];
    inBetween = [min(KErr), fliplr(max(KErr))];
    fill(Tc2, inBetween, 'g', EdgeColor='none');
    hold on
    plot(Tc,mean(KErr),LineWidth=2);
    hold off
    title("Our method")
    figure
    inBetween = [min(WErr), fliplr(max(WErr))];
    fill(Tc2, inBetween, 'g', EdgeColor='none');
    hold on
    plot(Tc,mean(WErr),LineWidth=2)
    hold off
    title("KDMD")
    
    % % Data storage for pgfplots
    temp = Tc.';
    save('DuffingTime.dat','temp','-ascii');
    temp = mean(KErr).';
    save('DuffingKMean.dat','temp','-ascii');
    temp = max(KErr).';
    save('DuffingKMax.dat','temp','-ascii');
    temp = min(KErr).';
    save('DuffingKMin.dat','temp','-ascii');
    temp = mean(WErr).';
    save('DuffingWMean.dat','temp','-ascii');
    temp = max(WErr).';
    save('DuffingWMax.dat','temp','-ascii');
    temp = min(WErr).';
    save('DuffingWMin.dat','temp','-ascii');
    if pinv
        temp = mean(WKDiff).';
        save('DuffingPinvDiffMean.dat','temp','-ascii');
        temp = min(WKDiff).';
        save('DuffingPinvDiffMin.dat','temp','-ascii');
        temp = max(WKDiff).';
        save('DuffingPinvDiffMax.dat','temp','-ascii');
    else
        temp = mean(WKDiff).';
        save('DuffingRegDiffMean.dat','temp','-ascii');
        temp = min(WKDiff).';
        save('DuffingRegDiffMin.dat','temp','-ascii');
        temp = max(WKDiff).';
        save('DuffingRegDiffMax.dat','temp','-ascii');
    end
    temp = Td.';
    save('DuffingTimeDiscrete.dat','temp','-ascii');
    temp = mean(KErrd).';
    save('DuffingKMeanDiscrete.dat','temp','-ascii');
    temp = max(KErrd).';
    save('DuffingKMaxDiscrete.dat','temp','-ascii');
    temp = min(KErrd).';
    save('DuffingKMinDiscrete.dat','temp','-ascii');
    temp = mean(WErrd).';
    save('DuffingWMeanDiscrete.dat','temp','-ascii');
    temp = max(WErrd).';
    save('DuffingWMaxDiscrete.dat','temp','-ascii');
    temp = min(WErrd).';
    save('DuffingWMinDiscrete.dat','temp','-ascii');
end