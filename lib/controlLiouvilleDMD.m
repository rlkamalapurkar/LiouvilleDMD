% This is an implementation of the control Liouville DMD method in 
% Rosenfeld and Kamalapurkar 2021, see
%
% https://arxiv.org/abs/2101.02620
%
% This function performs dynamic mode decomposition of a controlled 
% dynamical system from sampled state and control trajectories of the 
% system. 
%
% [Z,L,ef,r,f] = controlLiouvilleDMD(KT,K,X,U,t,mu) OR
% [Z,L,ef,r,f] = controlLiouvilleDMD(KT,K,X,U,t,mu,l)
%
% Inputs:
%    1) KT: svRKHS kernel 
%    2) K:  vvRKHS kernel 
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
%    1) Z:  Liouville modes (State dimension x number of modes)
%    2) L:  Eigenvalues (number of modes x 1)
%    3) ef: Eigenfunctions
%    4) r:  Reconstruction function
%    5) f:  Vector field
%
function [Z,L,ef,r,f] = controlLiouvilleDMD(KT,K,X,U,t,mu,varargin)

% Processing optional arguments and setting defaults
if nargin == 6
    l = 0; % default
elseif nargin == 7
    l = varargin{1};
elseif nargin > 7
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
S = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2));

% Control OCC Gram matrix and interaction matrix - M x M
G = zeros(M);
I = G;
% OCC Gram matrix and interaction matrix - M x M
GT=zeros(M);
IT=GT;
for i=1:M
    GT(:,i) = squeeze(pagemtimes(pagemtimes(S(:,1,i).',KT.K(X(:,:,i),X)),S));
    IT(:,i) = pagemtimes(KT.K(X(:,Lengths(i),i),X) - KT.K(X(:,1,i),X),S);

    G(:,i) = squeeze(pagemtimes(pagemtimes(S(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',U,'transpose'),4)),S));
    I(:,i) = squeeze(pagemtimes(pagemtimes(S(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',MU,'transpose'),4)),S));
end

% DMD
GT=GT+l*eye(size(GT)); % Regularization
G=G+l*eye(size(G)); % Regularization
[V,D] = eig(GT\(I*(G\IT.'))); % Eigendecomposition of the finite rank representation
L = diag(D); % Eigenvalues of the finite-rank representation
C = V./diag(sqrt(V'*GT*V)).'; % Normalized eigenvectors of finite rank representation

% Reconstruction
ef = @(x) C.'*squeeze(pagemtimes(KT.K(x,X),S)); % Eigenfunctions evaluated at x
IntMat = reshape(pagemtimes(X,S),n,M); % Integrals of trajectories
Z = IntMat/(C.'*GT); % Control Liouville modes
r = @(t,x) real(Z*(ef(x).*exp(L*t))); % Reconstruction function
f = @(x) real(Z*(L.*ef(x))); % vectorfield
end