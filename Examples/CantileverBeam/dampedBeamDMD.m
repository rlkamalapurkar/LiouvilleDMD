% This script generates snapshots of the position of mesh points in a
% cantilever beam and uses Liouville DMD to generate a predictive model
% from the data.
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
[resT,msh,longestPeriod] = cantileverBeamUndampedTransient(10,0.3);
clearvars -except resT msh longestPeriod
close all
addpath('../../lib');

%% Parameters
ShowVideo = 0;
SaveVideo = 0;

%% Kernel 
% (R2020b or newer version of MATLAB needed)
mu = 500; % 1000 for exponential, 0.0005 for linear
K = KernelRKHS('Linear',mu);
Regularization = 1e-10; % 3e-12 for exponential

%% Generate and format data for DMD
% Generate data
%[resT,msh,longestPeriod] = cantileverBeamUndampedTransient(10,0.3);
h = longestPeriod/100;
skip = 1;
% Segment single trajectory into multiple trajectories
Snapshots = zeros(1,2*size(msh.Nodes,1)*size(resT.Displacement.x(1:skip:end,:),1),size(resT.Displacement.x(1:skip:end,:),2));
Dimension = length(Snapshots(1,:,1));
TotalLength = length(Snapshots(1,1,:));
NumNodes = size(msh.Nodes(:,1:skip:end),2);
% Generate trajectories dataset and sample time matrix
TrajectoryLength = 31;
TotalTrajectories = TotalLength - TrajectoryLength + 1;
Trajectories = zeros(Dimension,TrajectoryLength,TotalTrajectories);
SampleTime = NaN*ones(TrajectoryLength,TotalTrajectories);
for j = 1:TotalTrajectories
    Trajectories(:,1:TrajectoryLength,j) = Snapshots(1,:,j:j+(TrajectoryLength-1));
    SampleTime(1:TrajectoryLength,j) = h*(0:1:TrajectoryLength-1);
end

%% Liouville DMD
[~,~,~,r,f] = LiouvilleDMD(K,Trajectories,SampleTime,[],Regularization);

%% Generalization
[V,~,~] = cantileverBeamUndampedTransient(15,0);
initialState = [V.Displacement.x(:,1);
                V.Displacement.y(:,1);
                zeros(2*NumNodes,1)];
Time = ((1:size(V.Displacement.x,2))-1)*h;
[~,indirectGeneralizationState] = ode45(@(t,x) f(x), Time, initialState);
supNorm = max(norm([V.Displacement.x(:); V.Displacement.y(:)]));
fig1 = figure();
indirectGeneralizationError = zeros(size(Time));
for i = 1:size(V.Displacement.x,2)
    indirectGeneralizationErrorNorm = norm([indirectGeneralizationState(1,1:NumNodes).' - V.Displacement.x(:,i),...
        indirectGeneralizationState(1,NumNodes+1:2*NumNodes).' - V.Displacement.y(:,i)]);
    indirectGeneralizationError(i) = indirectGeneralizationErrorNorm/supNorm;
end
plot(Time,indirectGeneralizationError,'LineWidth',1.5)
ylabel('Relative RMS Error','Interpreter','LaTeX');
xlabel('Time [s]','Interpreter','LaTeX');
set(gca,'fontsize',12,'TickLabelInterpreter','latex');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
filename = 'BeamDMDIndirectGeneralizationError';
%saveas(fig1,filename,'pdf');

%% Video
if ShowVideo || SaveVideo
    if SaveVideo
        filename = ['first-order-liouville-beam-reconstruction.mp4'];
        v = VideoWriter(filename, 'MPEG-4');
        v.FrameRate = 10;
        open(v);
    end
    fig1 = figure('units','pixels','position',[100 50 1300 850]);
    InitialState = Trajectories(1:end,1,1);
    for i = 1:150
        t = (i-1)*h;
        ZZ = r(t,InitialState);
        error_y = ZZ(NumNodes+1:2*NumNodes,1) - resT.Displacement.y(1:skip:end,i);
        error_x = ZZ(1:NumNodes,1) - resT.Displacement.x(1:skip:end,i);
        error_norm_pointwise = sqrt(error_x.^2 + error_y.^2);
        newNodes = zeros(size(msh.Nodes(:,1:skip:end)));
        newNodes(1,:) = msh.Nodes(1,1:skip:end)+resT.Displacement.x(1:skip:end,i).';
        newNodes(2,:) = msh.Nodes(2,1:skip:end)+resT.Displacement.y(1:skip:end,i).';
        pdeplot(newNodes,msh.Elements,"XYData",error_norm_pointwise,"ColorMap","parula")
        colorbar;
        clim([0 5e-4]);
        ylim([-0.02,0.12]);
        xlim([1,5]);
        drawnow;
        if SaveVideo
            frame = getframe(gcf);
            writeVideo(v,frame);
        end
    end
    if SaveVideo
        close(v);
    end
end