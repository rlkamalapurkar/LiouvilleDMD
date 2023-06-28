% This function performs provably convergent dynamic mode decomposition of 
% a dynamical system from sampled trajectories. For details, see
%
% https://arxiv.org/abs/2106.02639
%
% [Z,L,ef,r,f] = ConvergentLiouvilleEigenDMD(K,KT,X,T) OR
% [Z,L,ef,r,f] = ConvergentLiouvilleEigenDMD(K,KT,X,T,l)
%
% Inputs:
%  1,2) K,KT: Objects of the class 'Kernel', K.K is the kernel function
%          Example 1, exponential dot product:
%             K.K = @(X,Y) exp(1/mu*pagemtimes(X,'transpose',Y,'none'));
%          Example 2, Gaussian (NOT YET IMPLEMENTED):
%             K.K = @(X,Y) exp(-1/mu*(pagetranspose(sum(X.^2,1)) + ...
%             sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'))); 
%
%    3) X: A dataset of trajectories (3D array)
%          First dimension: State 
%          Second dimension: Time (size = length of longest trajectory)
%          Third dimension: Trajectory number
%
% *Trajectories can be of different lengths and irregularly sampled.*
% *The number of samples in each trajectory needs to be odd.*
% *Shorter trajectories need to be padded with zeros.*
%
%    4) t: A matrix of sample times
%          First dimension: Time (size = length of longest trajectory)
%          Second dimension: Trajectory number
%
% *Sample times for shorter trajectories need to be padded with NaNs.*
%
%    5) l: (Optional argument, default=0) Regularization coefficient
%          (needed if the Gram matrix is rank deficient)
%
% Outputs:
%    1) Z: Liouville modes (State dimension x number of modes)
%    2) L: Eigenvalues (number of modes x 1)
%    3) ef: Eigenfunction coefficients (number of modes x number of modes)
%    4) r: Reconstruction function
%    5) f: Approximation of the vector field
%
% © Rushikesh Kamalapurkar and Joel Rosenfeld
%
function [Z,L,ef,r,f] = ConvergentLiouvilleEigenDMD(K,KT,X,t,varargin)
% Processing optional arguments and setting defaults
if nargin == 4
    l = 0; % default
elseif nargin == 5
    l = varargin{1};
elseif nargin > 5
    error("Too many input arguments.")
end

% Check kernel parameter compatibility
mu = K.parameter;
muT = KT.parameter;
if mu < muT
    warning('Parameters incompatible with convergent DMD');
end

M = size(X,3); % Total number of trajectories
n = size(X,1); % State Dimension

% Store trajectory lengths for interaction matrix calculation
N = size(t,1)-sum(isnan(t));

% Simpsons rule weights
w = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2));

% Gram matrix and interaction matrix
G1=zeros(M);
G=zeros(M);
GT=zeros(M);
I=zeros(M);
for i=1:M
    G(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',K.K(X(:,:,i),X)),w));
    GT(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',KT.K(X(:,:,i),X)),w));
    if isequal(K.type,'Exponential') && isequal(KT.type,'Exponential')
        G1(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',KT.K((muT/mu)*X(:,:,i),(muT/mu)*X)),w));
    else
        error('Kernels other than the exponential kernel not implemented');
    end
    I(:,i) = pagemtimes(K.K(X(:,N(i),i),X) - K.K(X(:,1,i),X),w);
end

% DMD
G1 = G1 + l*eye(size(G1)); % Regularization
G = G + l*eye(size(G)); % Regularization
GT = GT + l*eye(size(GT)); % Regularization
[V,D] = eig(G1\(G*(GT\(I.')))); % Eigendecomposition of finite rank representation
C = V./diag(sqrt(V'*G*V)).'; % Normalized eigenvectors of the finite rank representation
IntMat = reshape(pagemtimes(X,w),n,M); % Integrals of trajectories
Z = IntMat/(C.'*G); % Liouville modes
L = diag(D); % Eigenvalues of the finite-rank representation

% Reconstruction
% Occupation kernels evaluated at x0: squeeze(pagemtimes(K(x0,W),S))
% Eigenfunctions evaluated at x:
ef = @(x) C.'*squeeze(pagemtimes(K.K(x,X),w));
% Reconstruction function:
r = @(t,x) Z*((C.'*squeeze(pagemtimes(K.K(x,X),w))).*exp(L*t)); 
% Vector field:
f = @(x) real(Z*((C.'*squeeze(pagemtimes(K.K(x,X),w))).*L));
end