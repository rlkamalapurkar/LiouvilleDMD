% This script generates trajectories of the Duffing oscillator and uses
% first order Koopman DMD to generate a predictive model from the data.
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function LorenzKoopmanDMD()
    rng(1)
    addpath('../../lib')

    % % Lorenz trajectory generator
    sigma = 10;
    beta = 8/3;
    rho = 28;
    xDot = @(t,x) [sigma*(x(2)-x(1)); x(1)*(rho-x(3))-x(2); x(1)*x(2)-beta*x(3)];
    n = 3;
    sampMin = -10;
    sampMax = 10;
    M = 100;
    IV_selection = 'grid'; 
    if strcmp(IV_selection,'random')
        % Get TotalTrajectories random IV's.
        x0 = sampMin + (sampMax - sampMin)*rand(n, M);
    elseif strcmp(IV_selection,'halton')
        % Get TotalTrajectories halton sequence
        haltonseq = @(n,d) net(haltonset(d),n);
        halton = haltonseq(M, n);
        x0 = sampMin + (sampMax - sampMin)*halton.';
    elseif strcmp(IV_selection,'grid')
        pointsPerDim = 6;
        x = linspace(sampMin,sampMax,pointsPerDim);
        [XX,YY,ZZ] = ndgrid(x);
        x0 = [XX(:) YY(:) ZZ(:)].';
        M = size(x0,2);
    else
        error('Unknown IV selection mode %s', IV_selection)
    end
    X=x0;
    Y=[];
    deltaT = 0.1;
    T = 0:deltaT:deltaT;
    noiseStandardDeviation = 0;
    for i=1:M
        [t,y]=ode45(@(t,x) xDot(t,x),T,x0(:,i));
        Y = [Y y(end,:).'+noiseStandardDeviation*randn(size(y(end,:).'))];
    end

    % % Kernel
    mu = 100000;
    l = 1e-6;
    K = KernelRKHS('Gaussian',mu);
    mu = 5000;
    lW = 1e-6;
    KW = KernelRKHS('Gaussian',mu);

    % % Koopman DMD
    [~,~,~,~,~,fc] = KoopmanDMD(X,Y,K,deltaT,RegTol=l);
    [~,~,~,~,~,fcW] = WilliamsKDMD(X,Y,KW,deltaT,lW);

    % % Reconstruction 
    tfRec = 5; % Final time for reconstruction
    Tc=0:0.1:tfRec; % Time vector for reconstruction
    XRec = [-7.5 + 15*rand(1,100);
        -7.5 + 15*rand(1,100);
        -7.5 + 15*rand(1,100)]; % Initial conditions for reconstruction
    occErr = zeros(100,numel(Tc));
    WErr = zeros(100,numel(Tc));
    for i=1:100
        x = XRec(:,i);
        % Actual trajectory
        [~,y]=ode45(@(t,x) xDot(t,x),Tc,x);
        maxNorm = max(vecnorm(y.'));
        % using continuous occupation kernel vector field
        [~,ycv]=ode45(@(t,x) fc(x),Tc,x);
        occErr(i,:) = vecnorm(y.'-ycv.')/maxNorm;
        % using continuous Williams vector field
        [~,ycW]=ode45(@(t,x) fcW(x),Tc,x);
        WErr(i,:) = vecnorm(y.'-ycW.')./maxNorm;
    end
    Tc2 = [Tc, fliplr(Tc)];
    inBetween = [min(occErr), fliplr(max(occErr))];
    fill(Tc2, inBetween, 'g', EdgeColor='none');
    hold on
    plot(Tc,mean(occErr),LineWidth=2);
    title("Occupation KDMD")
    hold off
    figure
    inBetween = [min(WErr), fliplr(max(WErr))];
    fill(Tc2, inBetween, 'g', EdgeColor='none');
    hold on
    plot(Tc,mean(WErr),LineWidth=2)
    title("Williams KDMD")
    hold off
    % save('LorenzTime.dat','Tc','-ascii');
    % temp = mean(occErr);
    % save('LorenzOccMean.dat','temp','-ascii');
    % temp = max(occErr);
    % save('LorenzOccMax.dat','temp','-ascii');
    % temp = min(occErr);
    % save('LorenzOccMin.dat','temp','-ascii');
    % temp = mean(WErr);
    % save('LorenzWMean.dat','temp','-ascii');
    % temp = max(WErr);
    % save('LorenzWMax.dat','temp','-ascii');
    % temp = min(WErr);
    % save('LorenzWMin.dat','temp','-ascii');
end