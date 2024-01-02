% This is an implementation of the control Liouville DMD method in 
% Rosenfeld and Kamalapurkar 2021, see
%
% https://arxiv.org/abs/2101.02620
%
% This function performs dynamic mode decomposition of a controlled 
% dynamical system from sampled state and control trajectories of the 
% system. 
%
% [Z,L,ef,r,f] = ControlLiouvilleDMD(KT,K,X,U,t,mu) OR
% [Z,L,ef,r,f] = ControlLiouvilleDMD(KT,K,X,U,t,mu,l)
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
%    7) RegTol:  (Optional argument, default=0) Regularization coefficient
%    8) PinvTol: (Optional argument, default=0) Pseudoinverse coefficient
%
% If RegTol and PinvTol are both nonzero or both zero, or if PinvTol is  
% supplied but RegTol is not, then pseudoinverse is used to invert Gram 
% matrices. If RegTol is supplied and PinvTol is not, then regularization 
% is used to invert Gram matrices.
%
% Outputs:
%    1) Z:  Liouville modes (State dimension x number of modes)
%    2) L:  Eigenvalues (number of modes x 1)
%    3) ef: Eigenfunctions
%    4) r:  Reconstruction function
%    5) f:  Vector field
%
function [Z,L,ef,r,f] = ControlLiouvilleDMD(KT,K,X,U,t,mu,NameValueArgs)

% Processing optional arguments and setting defaults
arguments
    KT
    K
    X double
    U double
    t double
    mu
    NameValueArgs.RegTol (1,1) {mustBeNumeric} = 0
    NameValueArgs.PinvTol (1,1) {mustBeNumeric} = 0
end
if NameValueArgs.RegTol ~= 0 && NameValueArgs.PinvTol ~=0
    warning('RegTol and PinvTol are both nonzero, defaulting to pseudoinverse. Call with RegTol ~= 0 and PinvTol = 0 to use regularization.');
end

M = size(X,3); % Number of trajectories
m = size(U,1); % Control dimension
n = size(X,1); % State dimension
maxN = size(X,2); % Number of samples in longest trajectory

% Values of the feedback controller
MU = mu(X);

% Concatenate control arrays with 1 in the first dimension
U = cat(1,ones(1,maxN,M),U(:,(1:maxN),:));
MU = cat(1,ones(1,maxN,M),MU(:,(1:maxN),:));

% Make 3D control arrays into 4D arrays for Gram matrix calculations
U = reshape(permute(U,[2,3,1]),maxN,1,M,m+1);
MU = reshape(permute(MU,[2,3,1]),maxN,1,M,m+1);

% Store trajectory lengths for interaction matrix calculation
N = size(t,1)-sum(isnan(t));

% Simpsons rule weights
w = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2));

% Control OCC Gram matrix and interaction matrix - M x M
G = zeros(M);
I = G;
% OCC Gram matrix and interaction matrix - M x M
GT=zeros(M);
IT=GT;
for i=1:M
    GT(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',KT.K(X(:,:,i),X)),w));
    IT(:,i) = pagemtimes(KT.K(X(:,N(i),i),X) - KT.K(X(:,1,i),X),w);

    G(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',U,'transpose'),4)),w));
    I(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',MU,'transpose'),4)),w));
end

% DMD
% Gram matrix inverse using regularization or pseudoinverse
if NameValueArgs.PinvTol == 0 && NameValueArgs.RegTol == 0
    GTInv = pinv(GT);
    GInv = pinv(G);
    FRR=GTInv*I*GInv*IT.';
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif NameValueArgs.PinvTol ~= 0
    GTInv = pinv(GT,NameValueArgs.PinvTol);
    GInv = pinv(G,NameValueArgs.PinvTol);
    FRR=GTInv*I*GInv*IT.';
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif NameValueArgs.RegTol ~= 0
    GT=GT+NameValueArgs.RegTol*eye(size(GT)); % Regularization
    G=G+NameValueArgs.RegTol*eye(size(G)); % Regularization
    FRR = GT\(I*(G\IT.'));
    % disp('Inverting Gram matrices using regularization.')
end

[V,D] = eig(FRR); % Eigendecomposition of the finite rank representation
L = diag(D); % Eigenvalues of the finite-rank representation
C = V./diag(sqrt(V'*GT*V)).'; % Normalized eigenvectors of finite rank representation

% Reconstruction
ef = @(x) C.'*squeeze(pagemtimes(KT.K(x,X),w)); % Eigenfunctions evaluated at x
IntMat = reshape(pagemtimes(X,w),n,M); % Integrals of trajectories

if NameValueArgs.PinvTol == 0 && NameValueArgs.RegTol == 0
    CGTInv = pinv(C.'*GT);
    Z = IntMat*CGTInv; % Control Liouville modes
elseif NameValueArgs.PinvTol ~= 0
    CGTInv = pinv(C.'*GT,NameValueArgs.PinvTol);
    Z = IntMat*CGTInv; % Control Liouville modes
elseif NameValueArgs.RegTol ~= 0
    Z = IntMat/(C.'*GT); % Control Liouville modes
end

r = @(t,x) real(Z*(ef(x).*exp(L*t))); % Reconstruction function
f = @(x) real(Z*(L.*ef(x))); % vectorfield
end