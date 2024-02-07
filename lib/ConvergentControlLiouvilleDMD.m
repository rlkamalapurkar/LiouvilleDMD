% This is an implementation of the convergent control Liouville DMD 
%
% This function performs dynamic mode decomposition of a controlled 
% dynamical system from sampled state and control trajectories of the 
% system. 
%
% [Z,S,lsf,rsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu) OR
% [Z,S,lsf,rsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu,RegTol=l) OR
% [Z,S,lsf,rsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu,PinvTol=l)
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
%    7) RegTol:  (Optional argument, default=0) Regularization coefficient
%    8) PinvTol: (Optional argument, default=0) Pseudoinverse coefficient
%
% If RegTol and PinvTol are both nonzero or both zero, or if PinvTol is  
% supplied but RegTol is not, then pseudoinverse is used to invert Gram 
% matrices. If RegTol is supplied and PinvTol is not, then regularization 
% is used to invert Gram matrices.
%
% Outputs:
%    1) Z  :  Liouville modes (State dimension x number of modes)
%    2) S  :  Singular Values (number of modes x 1)
%    3) lsf:  Left singular functions
%    4) rsf:  Right singular functions
%    5) f  :  Vector field
%
function [Z,S,rsf,lsf,f] = ConvergentControlLiouvilleDMD(Kd,Kr,K,X,U,t,mu,NameValueArgs)

% Processing optional arguments and setting defaults
arguments
    Kd
    Kr
    K
    X double
    U double
    t double
    mu
    NameValueArgs.RegTol (1,1) {mustBeNumeric} = NaN
    NameValueArgs.PinvTol (1,1) {mustBeNumeric} = NaN
end
if ~isnan(NameValueArgs.RegTol) && ~isnan(NameValueArgs.PinvTol)
    warning('RegTol and PinvTol are both nonzero, defaulting to pseudoinverse. Call with RegTol ~= 0 and PinvTol = 0 to use regularization.');
    NameValueArgs.RegTol = NaN;
end

M = size(X,3); % Number of trajectories
m = size(U,1); % Control dimension
n = size(X,1); % State dimension
maxN = size(X,2); % Number of samples in longest trajectory
N = size(t,1)-sum(isnan(t)); % Trajectory lengths
w = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2)); % Simpsons rule weights

MU = mu(X); % Values of the feedback controller

% Concatenate control arrays with 1 in the first dimension
U = cat(1,ones(1,maxN,M),U(:,(1:maxN),:));
MU = cat(1,ones(1,maxN,M),MU(:,(1:maxN),:));

% Make 3D control arrays into 4D arrays for Gram matrix calculations
U = reshape(permute(U,[2,3,1]),maxN,1,M,m+1);
MU = reshape(permute(MU,[2,3,1]),maxN,1,M,m+1);

% Control OCC Gram matrix and interaction matrix - M x M
Gb = zeros(M);
I = Gb;
% OCC Gram matrix and interaction matrix - M x M
Gr=zeros(M);
D=zeros(n,M);
for i=1:M
    Gr(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',Kr.K(X(:,:,i),X)),w));
    D(:,i) = X(:,N(i),i) - X(:,1,i);

    Gb(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',U,'transpose'),4)),w));
    I(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',MU,'transpose'),4)),w));
end

% DMD
if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
    GrInv = pinv(Gr);
    GbInv = pinv(Gb);
    FRR=GrInv*I*GbInv;
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.PinvTol)
    GrInv = pinv(Gr,NameValueArgs.PinvTol);
    GbInv = pinv(Gb,NameValueArgs.PinvTol);
    FRR=GrInv*I*GbInv;
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.RegTol)
    Gr=Gr+NameValueArgs.RegTol*eye(size(Gr)); % Regularization
    Gb=Gb+NameValueArgs.RegTol*eye(size(Gb)); % Regularization
    FRR=Gr\(I/Gb);
    % disp('Inverting Gram matrices using regularization.')
end

[W,S,V] = svd(FRR); % SVD of the finite rank representation
Z = D*V; % Liouville modes
rsf = @(x) W.'*squeeze(pagemtimes(Kr.K(x,X),w)); % Right singular functions evaluated at x
lsf = @(x) V.'*(arrayfun(@(l) Kd.K(x,X(:,N(l),l)),(1:M).') ...
    - arrayfun(@(l) Kd.K(x,X(:,1,l)),(1:M).')); % Left singular functions evaluated at x

% SysID
temp=D*FRR.';
f = @(x) real(temp*squeeze(pagemtimes(Kr.K(x,X),w))); % Vector field
end