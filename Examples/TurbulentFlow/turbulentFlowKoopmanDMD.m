% This script uses finite element snapshots from the dataset 7999.am at
% https://doi.org/10.3929/ethz-b-000515488
% and uses Koopman DMD to generate a predictive model from the data.
% Uses the function LoadData_Amira.m to read 7999.am
%
% © Rushikesh Kamalapurkar and Moad Abudia
%
function turbulentFlowKoopmanDMD()
%% Set up paths
DATAPATH = '../../../DATA';
addpath("../../lib")
[Header,Data] = LoadData_Amira([DATAPATH '/7999.am']);
X = reshape(Data,Header.xSize*Header.ySize,Header.zSize,2);
X = [X(:,:,1);X(:,:,2)];

normalizationFactor = max(vecnorm(X));
W = X(:,1:300)/normalizationFactor;
V = X(:,2:301)/normalizationFactor;

deltaT = Header.SliceThickness;

%% Kernel selection
mu = 0.75;
l = 0.0001;
K = KernelRKHS('Gaussian',mu); 

muW = 0.000518;
KW = KernelRKHS('Gaussian',muW); 

%% Kernel DMD
[~,~,~,~,dr,~] = KoopmanDMD(W,V,K,deltaT,RegTol=l);
[~,~,~,~,drW,~] = WilliamsKDMD(W,V,KW,deltaT);

%% Plots for the paper
reconstructionError = zeros(Header.zSize-1,1);
reconstructionErrorW = zeros(Header.zSize-1,1);
time = zeros(Header.zSize-1,1);
x = X(:,1)/normalizationFactor;
for i=1:numel(reconstructionError)
    time(i) = deltaT*i;
    reconstructionError(i) = norm(X(:,1+i)/normalizationFactor - dr(i,x));
    reconstructionErrorW(i) = norm(X(:,1+i)/normalizationFactor - drW(i,x));
end
plot(time,reconstructionError);hold on;plot(time,reconstructionErrorW);hold off;
legend("Gonzalez et al.","Williams et al.");
xlabel("t [s]");
ylabel("Norm of the relative reconstruction error")
reconstructionError = [time reconstructionError];
save('KoopmanError.dat','reconstructionError','-ascii');
reconstructionErrorW = [time reconstructionErrorW];
save('WilliamsError.dat','reconstructionErrorW','-ascii');

%% Animation
anim=1;
if anim
    [XX YY] = meshgrid(0:0.002:0.002*511);
    index = 0;
    for i=250:350
        t = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
        nexttile
        h1=surf(XX,YY,Data(:,:,i+1,1));
        set(h1,'edgecolor','none');
        xlim([0,1]);
        ylim([0,1]);
        axis equal
        view(0,90);
        colorbar;
        clim([-0.8,0.8]);
        title('True','FontSize',12,'Interpreter','latex','Units', 'normalized', 'Position', [0.5, 0.8, 0]);
        set(gca,'YTickLabel',[],'XTickLabel',[],'XColor', 'none','YColor','none');
        grid off
        
        nexttile
        snapshot = dr(i,x);
        xVel = snapshot(1:512*512);
        xVel = reshape(xVel,512,512);
        h2=surf(XX,YY,xVel*normalizationFactor);
        set(h2,'edgecolor','none');
        xlim([0,1]);
        ylim([0,1]);
        axis equal
        view(0,90);
        colorbar;
        clim([-0.8,0.8]);
        title('Estimated','FontSize',12,'Interpreter','latex','Units', 'normalized', 'Position', [0.5, 0.8, 0]);
        set(gca,'YTickLabel',[],'XTickLabel',[],'XColor', 'none','YColor','none');
        grid off
        filename = ['image-' num2str(index) '.svg'];
        t.Children(1).Color="none";
        t.Children(2).Color="none";
        %exportgraphics(t,filename, 'ContentType', 'vector', 'BackgroundColor', 'none')
        drawnow;
        saveas(gcf,filename);
        index = index+1;
    end
end
end