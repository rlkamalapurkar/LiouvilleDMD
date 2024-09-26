% X = [reshape(u,1200,236691) reshape(v,1200,236691) reshape(w,1200,236691)].';
% X = X(:,121:end);
clear all
addpath('../../lib')
DATAPATH = '../../../DATA';
load([DATAPATH '/BuildingFlowData.mat']);
normalizationFactor = max(vecnorm(X));
W = X(:,121:720)/normalizationFactor;
V = X(:,122:721)/normalizationFactor;
deltaT = 1;
mu = 0.00001;
K = KernelRKHS('Gaussian',mu); 
[~,~,~,~,dr,~] = KoopmanDMD(W,V,K,deltaT);
[~,~,~,~,drW,~] = WilliamsKDMD(W,V,K,deltaT);
reconstructionError = zeros(numel(121:1000),1);
reconstructionErrorW = zeros(numel(121:1000),1);
x = X(:,121)/normalizationFactor;
for i=1:numel(reconstructionError)
    reconstructionError(i) = norm(X(:,121+i)/normalizationFactor - dr(i,x));
    reconstructionErrorW(i) = norm(X(:,121+i)/normalizationFactor - drW(i,x));
end
plot(reconstructionError);
hold on;plot(reconstructionErrorW);hold off;

%% Animation
anim=1;
if anim
    [XX YY] = meshgrid(0:0.002:0.002*511);
    figure;
    for i=250:350
        subplot(1,2,1)
        title(['snapshot ' num2str(i)])
        h1=surf(XX,YY,Data(:,:,i+1,1));
        set(h1,'edgecolor','none');
        view(0,90);
        colorbar;
        subplot(1,2,2)
        title(['snapshot ' num2str(i)])
        snapshot = dr(i,x);
        xVel = snapshot(1:512*512);
        xVel = reshape(xVel,512,512);
        h2=surf(XX,YY,xVel*normalizationFactor);
        set(h2,'edgecolor','none');
        view(0,90);
        colorbar;
        drawnow;
    end
end