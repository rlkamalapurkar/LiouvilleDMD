% This script generates snapshots of the position of mesh points in an
% undamped cantilever beam and uses second order LiouvilleDMD to generate a 
% predictive model from the data.
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
clear all
close all
addpath('../../lib');

%% Parameters
ShowVideo = 0; % Show reconstruction animation
SaveVideo = 0; % Save animation as mp4 video
PrintReconstruction = 1; % Show reconstruction snapshots
IndirectReconstruction = 1;
PrintError = 1;

%% Kernel 
% (R2020b or newer version of MATLAB needed)
mu = 0.0005; % 0.08 for exponential, 0.0005 for linear
K = KernelRKHS('Linear',mu);
Regularization = 1e-14;

%% Generate and format data for DMD
% Generate data
% See https://www.mathworks.com/help/pde/ug/dynamics-of-a-damped-cantilever-beam.html
% MATLAB Partial Differential Equations Toolbox required
[resT,msh,longestPeriod] = cantileverBeamUndampedTransient(10,0);
h = longestPeriod/100;

% Segment single trajectory into multiple trajectories
Snapshots = zeros(1,size(msh.Nodes,1)*size(resT.Displacement.x,1),size(resT.Displacement.x,2));
Snapshots(1,:,:) = [resT.Displacement.x;resT.Displacement.y];
Velocities = zeros(1,size(msh.Nodes(1,:),1)*size(resT.Displacement.x,1),size(resT.Displacement.x,2));
Dimension = length(Snapshots(1,:,1));
TotalLength = length(Snapshots(1,1,:));
NumNodes = size(msh.Nodes,2);
% Generate trajectories dataset and sample time matrix
TrajectoryLength = 31;
TotalTrajectories = TotalLength - TrajectoryLength + 1;
Trajectories = zeros(Dimension,TrajectoryLength,TotalTrajectories);
Derivatives = zeros(Dimension,TotalTrajectories);
SampleTime = NaN*ones(TrajectoryLength,TotalTrajectories);
for j = 1:TotalTrajectories
    Trajectories(:,1:TrajectoryLength,j) = Snapshots(1,:,j:j+(TrajectoryLength-1));
    Derivatives(:,j) = Velocities(:,j);
    SampleTime(1:TrajectoryLength,j) = h*(0:1:TrajectoryLength-1);
end

%% Liouville DMD
[~,~,~,directReconstruct,vectorField] = SecondOrderLiouvilleDMD(K,Trajectories,SampleTime,[],Regularization);

ll = [0,0.006,0.011,0.019,0.0275,0.037,0.045,0.0555,0.064,0.073,0.0825,0.091,0.1];
hl = [0,0.01,0.018,0.0275,0.036,0.044,0.0551,0.063,0.0722,0.0824,0.0905,0.095,0.1];
beamWidth = numel(ll);
beamLength = 500;
[YY,II] = sort(msh.Nodes(2,:));
indexArray = zeros(1,beamLength*beamWidth);
for i=1:beamWidth
    temp = find((YY <= hl(i)) .* (YY >= ll(i)));
    indexArray((i-1)*beamLength+1:i*beamLength) = temp(1:beamLength);
end
II = II(indexArray);
%% Video
if ShowVideo || SaveVideo
    if SaveVideo
        filename = ['second-order-liouville-beam-reconstruction.mp4'];
        v = VideoWriter(filename, 'MPEG-4');
        v.FrameRate = 10;
        open(v);
    end
    figure('units','pixels','position',[100 100 1300 1000]);
    for i = 1:150
        t = (i-1)*h;
        ZZ = real(directReconstruct(t,Trajectories(1:end,1,1),Derivatives(:,1)));
        error_y = ZZ(NumNodes+1:end) - resT.Displacement.y(:,i);
        error_x = ZZ(1:NumNodes) - resT.Displacement.x(:,i);
        error_norm_pointwise = sqrt(error_x.^2 + error_y.^2);
        newNodes = zeros(size(msh.Nodes));
        newNodes(1,:) = msh.Nodes(1,:)+resT.Displacement.x(:,i).';
        newNodes(2,:) = msh.Nodes(2,:)+resT.Displacement.y(:,i).';
        pdeplot(newNodes,msh.Elements,"XYData",error_norm_pointwise,"ColorMap","parula")
        colorbar;
        caxis([0 2.5e-3]);
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
%% Print reconstruction
if PrintReconstruction
    ReconstructionIndices = [1 30 70 100 130];
    fig1 = figure();
    for ii=1:numel(ReconstructionIndices)
        i = ReconstructionIndices(ii);
        t = (i-1)*h;
        ZZ = real(directReconstruct(t,Trajectories(1:end,1,1),Derivatives(:,1)));
        error_y = ZZ(NumNodes+1:end) - resT.Displacement.y(:,i);
        error_x = ZZ(1:NumNodes) - resT.Displacement.x(:,i);
        error_norm_pointwise = sqrt(error_x.^2 + error_y.^2);
        subplot('position',[0 0.95*(ii-1)/numel(ReconstructionIndices) 0.8 0.95/numel(ReconstructionIndices)],'parent',fig1);
        newNodes = zeros(size(msh.Nodes));
        newNodes(1,:) = msh.Nodes(1,:)+resT.Displacement.x(:,i).';
        newNodes(2,:) = msh.Nodes(2,:)+resT.Displacement.y(:,i).';
        pdeplot(newNodes,msh.Elements,"XYData",error_norm_pointwise,"ColorMap","parula","ColorBar","off")
        caxis([0 2.5e-3]);
        ylim([-0.02,0.12]);
        xlim([1,5]);
        set(gca,'YTickLabel',[],'XTickLabel',[]);
        title({'t =', [num2str(t*1000,2) ' ms']},'FontSize',11,'fontweight','normal','Units','normalized','Position',[1.06, 0.3, 0],'Interpreter','LaTeX');
    end
    ax1 = axes(fig1,'visible','off');
    c = colorbar(ax1,'Position',[0.9 0.1 0.022 0.8],'TickLabelInterpreter','latex');
    caxis([0 2.5e-3]);
    set(ax1,'fontsize',12,'TickLabelInterpreter','latex');
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [7 5]);
    set(gcf, 'PaperPosition', [0 0 7 5]);
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    fig1.Renderer='painters';
    filename = 'secondOrderLiouvilleBeamReconstruction';
    saveas(fig1,filename,'pdf');
end
%% Print error
if PrintError
    supNorm = max(norm([resT.Displacement.x(:); resT.Displacement.y(:)]));
    fig1 = figure();
    Time = ((1:size(resT.Displacement.x,2))-1)*h;
    relativeError = zeros(size(Time));
    for i = 1:size(resT.Displacement.x,2)
        t = (i-1)*h;
        ZZ = real(directReconstruct(t,Trajectories(1:end,1,1),Derivatives(:,1)));
        error_y = ZZ(NumNodes+1:end) - resT.Displacement.y(:,i);
        error_x = ZZ(1:NumNodes) - resT.Displacement.x(:,i);
        error_norm = norm([error_x error_y]);
        relativeError(i) = error_norm/supNorm;
    end
    plot(Time,relativeError,'LineWidth',2)
    ylabel('RMS Error','Interpreter','LaTeX');
    xlabel('Time [s]','Interpreter','LaTeX');
    set(gca,'fontsize',12,'TickLabelInterpreter','latex');
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [6 5]);
    set(gcf, 'PaperPosition', [0 0 6 5]);
    filename = 'secondOrderLiouvilleBeamReconstructionError';
    saveas(fig1,filename,'pdf');
end
%% Indirect reconstruction
odefun = @(t,x) [x(2*NumNodes+1:end); real(vectorField(x(1:2*NumNodes)))];
if IndirectReconstruction
    InitialState = [Trajectories(1:end,1,1);Derivatives(:,1)];
    Time = ((1:size(resT.Displacement.x,2))-1)*h;
    [~,ZZ_indirect] = ode45(@(t,x) odefun(t,x), Time, InitialState);
    if ShowVideo || SaveVideo
        figure('units','pixels','position',[100 100 1300 1000]);
        for i = 1:150
            error_y = ZZ_indirect(i,NumNodes+1:2*NumNodes).' - resT.Displacement.y(:,i);
            error_x = ZZ_indirect(i,1:NumNodes).' - resT.Displacement.x(:,i);
            pointwise_error_norm = sqrt(error_x.^2 + error_y.^2);
            newNodes = zeros(size(msh.Nodes));
            newNodes(1,:) = msh.Nodes(1,:)+resT.Displacement.x(:,i).';
            newNodes(2,:) = msh.Nodes(2,:)+resT.Displacement.y(:,i).';
            pdeplot(newNodes,msh.Elements,"XYData",pointwise_error_norm,"ColorMap","parula")
            colorbar;
            caxis([0 2.5e-3]);
            ylim([-0.02,0.12]);
            xlim([1,5]);
            drawnow;
        end
    end
    if PrintError
        supNorm = max(norm([resT.Displacement.x(:); resT.Displacement.y(:)]));
        fig1 = figure();
        relativeError = zeros(size(Time));
        for i = 1:size(resT.Displacement.x,2)
            error_y = ZZ_indirect(i,NumNodes+1:2*NumNodes).' - resT.Displacement.y(:,i);
            error_x = ZZ_indirect(i,1:NumNodes).' - resT.Displacement.x(:,i);
            error_norm = norm([error_x error_y]);
            relativeError(i) = error_norm/supNorm;
        end
        plot(Time,relativeError,'LineWidth',2)
        ylabel('RMS Error','Interpreter','LaTeX');
        xlabel('Time [s]','Interpreter','LaTeX');
        set(gca,'fontsize',12,'TickLabelInterpreter','latex');
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperUnits', 'inches');
        set(gcf, 'PaperSize', [6 5]);
        set(gcf, 'PaperPosition', [0 0 6 5]);
        filename = 'secondOrderLiouvilleBeamReconstructionErrorIndirect';
        saveas(fig1,filename,'pdf');
    end
end
%% Generalization
[resTValidation,~,~] = cantileverBeamUndampedTransient(15,0);
InitialState = [resTValidation.Displacement.x(:,1);resTValidation.Displacement.y(:,1);zeros(2*NumNodes,1)];
Time = ((1:size(resTValidation.Displacement.x,2))-1)*h;
[~,ZZ_indirect_validation] = ode45(@(t,x) odefun(t,x), Time, InitialState);
supNorm = max(norm([resTValidation.Displacement.x(:); resTValidation.Displacement.y(:)]));
fig1 = figure();
relativeError = zeros(size(Time));
relativeErrorIndirect = zeros(size(Time));
for i = 1:size(resTValidation.Displacement.x,2)
    error_y = ZZ_indirect_validation(i,NumNodes+1:2*NumNodes).' - resTValidation.Displacement.y(:,i);
    error_x = ZZ_indirect_validation(i,1:NumNodes).' - resTValidation.Displacement.x(:,i);
    error_norm = norm([error_x error_y]);
    relativeErrorIndirect(i) = error_norm/supNorm;
    t = (i-1)*h;
    ZZ = real(directReconstruct(t,[resTValidation.Displacement.x(:,1);resTValidation.Displacement.y(:,1)],zeros(2*NumNodes,1)));
    error_y = ZZ(NumNodes+1:end) - resTValidation.Displacement.y(:,i);
    error_x = ZZ(1:NumNodes) - resTValidation.Displacement.x(:,i);
    error_norm = norm([error_x error_y]);
    relativeError(i) = error_norm/supNorm;
end
plot(Time,relativeError,Time,relativeErrorIndirect,'LineWidth',1.5)
ylabel('Relative RMS Error','Interpreter','LaTeX');
xlabel('Time [s]','Interpreter','LaTeX');
l=legend('Direct','Indirect');
set(l,'Interpreter','latex');
set(gca,'fontsize',12,'TickLabelInterpreter','latex');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 5]);
set(gcf, 'PaperPosition', [0 0 6 5]);
filename = 'secondOrderLiouvilleBeamGeneralizationError';
saveas(fig1,filename,'pdf');