% This is an implementation of the convergent control Liouville DMD 
%
% This function performs dynamic mode decomposition of a controlled 
% dynamical system from sampled state and control trajectories of the 
% system. 
%
% [Z,L,lsf,rsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu) OR
% [Z,L,lsf,rsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu,l)
%
% Inputs:
%    1) Kd: domain svRKHS kernel 
%    2) K:  vvRKHS kernel
%    3) Kr: range svRKHS kernel 
%    3) X:  A dataset of state trajectories
%           First dimension: State
%           Second dimension: Time (size = length of longest trajectory)
%           Third dimension: Trajectory number
%    4) U:  A dataset of control trajectories
%           First dimension: Control
%           Second dimension: Time (size = length of longest trajectory)
%           Third dimension: Trajectory number
%    5) t:  A matrix of sample times
%           First dimension: Time (size = length of longest trajectory)
%           Second dimension: Trajectory number
%    6) mu: A feedback control function compatible with 3D arrays
%    7) l:  (Optional argument, default=0) Regularization coefficient
%
% Outputs:
%    1) Z  :  Liouville modes (State dimension x number of modes)
%    2) S  :  Singular Values (number of modes x 1)
%    3) lsf: Left singular functions
%    4) rsf: Right singular functions
%    5) f  :  Vector field
%
function [Z,S,rsf,lsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu,varargin)

% Processing optional arguments and setting defaults
if nargin == 7
    l = 0; % default
elseif nargin == 8
    l = varargin{1};
elseif nargin > 8
    error("Too many input arguments.")
end

M = size(X,3); % Number of trajectories
m = size(U,1); % Control dimension
n = size(X,1); % State dimension
N = size(X,2); % Number of samples in longest trajectory

% Values of the feedback controller
MU = mu(X);

% Concatenate control arrays with 1 in the first dimension
U = cat(1,ones(1,N,M),U(:,(1:N),:));
MU = cat(1,ones(1,N,M),MU(:,(1:N),:));

% Make 3D control arrays into 4D arrays for Gram matrix calculations
U = reshape(permute(U,[2,3,1]),N,1,M,m+1);
MU = reshape(permute(MU,[2,3,1]),N,1,M,m+1);

% Store trajectory lengths for interaction matrix calculation
Lengths = size(t,1)-sum(isnan(t));

% Simpsons rule weights
w = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2));

% Control OCC Gram matrix and interaction matrix - M x M
Gb = zeros(M);
I = Gb;
% OCC Gram matrix and interaction matrix - M x M
Gr=zeros(M);
D=zeros(n,M);
for i=1:M
    Gr(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',Kr.K(X(:,:,i),X)),w));
    D(:,i) = X(:,Lengths(i),i) - X(:,1,i);

    Gb(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',U,'transpose'),4)),w));
    I(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',MU,'transpose'),4)),w));
end

% DMD
Gr=Gr+l*eye(size(Gr)); % Regularization
Gb=Gb+l*eye(size(Gb)); % Regularization
FRR=Gr\(I/Gb);
[W,S,V] = svd(FRR); % SVD of the finite rank representation
Z = D*V; % Liouville modes

% Occupation kernels evaluated at x: squeeze(pagemtimes(K(x,W),S))
% Right singular functions evaluated at x:
rsf = @(x) W.'*squeeze(pagemtimes(Kr.K(x,X),w));
% Left singular functions evaluated at x:
lsf = @(x) V.'*(arrayfun(@(l) Kd.K(x,X(:,Lengths(l),l)),(1:M).') ...
    - arrayfun(@(l) Kd.K(x,X(:,1,l)),(1:M).'));
% Reconstruction
f = @(x) real(D*FRR.'*squeeze(pagemtimes(Kr.K(x,X),w))); % vectorfield
end