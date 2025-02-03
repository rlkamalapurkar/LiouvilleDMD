% This is an implementation of the convergent control Liouville DMD 
%
% This function performs dynamic mode decomposition of a controlled 
% dynamical system from sampled state and control trajectories of the 
% system. 
%
% [Z,S,lsf,rsf,fg] = ConvergentControlLiouvilleDMDOpenLoop(Kd,K,X,U,t) OR
% [Z,S,lsf,rsf,fg] = ConvergentControlLiouvilleDMDOpenLoop(Kd,K,X,U,t,RegTol=l) OR
% [Z,S,lsf,rsf,fg] = ConvergentControlLiouvilleDMDOpenLoop(Kd,K,X,U,t,PinvTol=l)
%
% Inputs:
%    1) Kd: domain svRKHS kernel 
%    2) K:  vvRKHS kernel
%    2) X:  A dataset of state trajectories
%           First dimension: State
%           Second dimension: Time (size = length of longest trajectory)
%           Third dimension: Trajectory number
%    3) U:  A dataset of control trajectories
%           First dimension: Control
%           Second dimension: Time (size = length of longest trajectory)
%           Third dimension: Trajectory number
%    4) t:  A matrix of sample times
%           First dimension: Time (size = length of longest trajectory)
%           Second dimension: Trajectory number
%    5) RegTol:  (Optional argument, default=0) Regularization coefficient
%    6) PinvTol: (Optional argument, default=0) Pseudoinverse coefficient
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
%    5) fg :  function that maps x to [f(x) g(x)]
%
function [Z,S,rsf,lsf,fg] = ConvergentControlLiouvilleDMDOpenLoop(Kd,K,X,U,t,NameValueArgs)

% Processing optional arguments and setting defaults
arguments
    Kd
    K
    X double
    U double
    t double
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

% Concatenate control array with 1 in the first dimension
U = cat(1,ones(1,maxN,M),U(:,(1:maxN),:));

% Make 3D control array into 4D array for Gram matrix calculations
U = reshape(permute(U,[2,3,1]),maxN,1,M,m+1);

% Control OCC Gram matrix and interaction matrix - M x M
Gb = zeros(M);
% Kernel difference matrix - M x M
D=zeros(n,M);
for i=1:M
    D(:,i) = X(:,N(i),i) - X(:,1,i);
    Gb(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',U,'transpose'),4)),w));
end

% DMD
if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
    GbInv = pinv(Gb);
    FRR=GbInv;
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.PinvTol)
    GbInv = pinv(Gb,NameValueArgs.PinvTol);
    FRR=GbInv;
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.RegTol)
    Gb=Gb+NameValueArgs.RegTol*eye(size(Gb)); % Regularization
    FRR=pinv(Gb);
    % disp('Inverting Gram matrices using regularization.')
end

[W,S,V] = svd(FRR); % SVD of the finite rank representation
Z = D*V; % Liouville modes
rsf = @(x) W.'*squeeze(pagemtimes(K.K(x,X).*permute(U,[2,1,3,4]),w)); % Right singular functions evaluated at x
lsf = @(x) V.'*(arrayfun(@(l) Kd.K(x,X(:,N(l),l)),(1:M).') ...
    - arrayfun(@(l) Kd.K(x,X(:,1,l)),(1:M).')); % Left singular functions evaluated at x

% SysID
temp=D*FRR.';
fg = @(x) temp*squeeze(pagemtimes(K.K(x,X).*permute(U,[2,1,3,4]),w)); % Vector field
end