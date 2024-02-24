% This function performs provably convergent dynamic mode decomposition of 
% a dynamical system from sampled trajectories.
%
% [Z,S,lsf,rsf,f] = ConvergentLiouvilleDMD(Kd,Kr,X,T) OR
% [Z,S,lsf,rsf,f] = ConvergentLiouvilleDMD(Kd,Kr,X,T,RegTol=l) OR
% [Z,S,lsf,rsf,f] = ConvergentLiouvilleDMD(Kd,Kr,X,T,PinvTol=l)
%
% Inputs:
%  1,2) Kd,Kr: Objects of the class 'KernelRKHS' for the domain and the
%          range. It is assumed that Kr.K, Kd.K are the kernel functions
%          Example 1, exponential dot product:
%             Kr.K = @(X,Y) exp(1/mu*pagemtimes(X,'transpose',Y,'none'));
%          Example 2, Gaussian (NOT YET IMPLEMENTED):
%             Kr.K = @(X,Y) exp(-1/mu*(pagetranspose(sum(X.^2,1)) + ...
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
%    4) T: A matrix of sample times
%          First dimension: Time (size = length of longest trajectory)
%          Second dimension: Trajectory number
%
% *Sample times for shorter trajectories need to be padded with NaNs.*
%
%    5) l: (Optional argument, default=0) Regularization coefficient or
%          pseudoinverse truncation tolerance. By default, the Gram matrix
%          is inverted using MATLAB's pinv function. If PinvTol is
%          supplied, it is passed to pinv as tolerance. If RegTol is set
%          instead, then the Gram matrix G is regularized as G = G + lI
%          before it is inverted.
%
% Outputs:
%    1) Z: Liouville modes (State dimension x number of modes)
%    2) S: Singular values (number of modes x number of modes)
%    3) lsf: Left singular functions
%    4) rsf: Right singular functions
%    4) f: Approximation of the vector field
%
% © Rushikesh Kamalapurkar
%
function [Z,S,lsf,rsf,f] = ConvergentLiouvilleDMD(Kd,Kr,X,t,NameValueArgs)
% Process optional arguments and set defaults
arguments
    Kd
    Kr
    X double
    t double
    NameValueArgs.RegTol (1,1) {mustBeNumeric} = NaN
    NameValueArgs.PinvTol (1,1) {mustBeNumeric} = NaN
end
if ~isnan(NameValueArgs.RegTol) && ~isnan(NameValueArgs.PinvTol)
    warning('RegTol and PinvTol are both nonzero, defaulting to pseudoinverse. Call with RegTol ~= 0 and PinvTol = 0 to use regularization.');
    NameValueArgs.RegTol = NaN;
end
    
M = size(X,3); % Total number of trajectories
n = size(X,1); % State Dimension
N = size(t,1)-sum(isnan(t)); % Trajectory lengths
w = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2)); % Simpsons rule weights

% Gram matrix and kernel difference matrix
Gr=zeros(M);
D = zeros(n,M);
for i=1:M
    Gr(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',Kr.K(X(:,:,i),X)),w));
    D(:,i) = X(:,N(i),i) - X(:,1,i);
end

% DMD
if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
    GrInv = pinv(Gr);
    disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.PinvTol)
    GrInv = pinv(Gr,NameValueArgs.PinvTol);
    disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.RegTol)
    GrInv = inv(Gr + NameValueArgs.RegTol*eye(size(Gr)));
    disp('Inverting Gram matrices using regularization.')
end

[W,S,V] = svd(GrInv); % SVD of the inverse of Gr
Z = D*V; % Liouville modes
rsf = @(x) W.'*squeeze(pagemtimes(Kr.K(x,X),w)); % Right singular functions evaluated at x
lsf = @(x) V.'*(arrayfun(@(l) Kd.K(x,X(:,N(l),l)),(1:M).') ...
    - arrayfun(@(l) Kd.K(x,X(:,1,l)),(1:M).')); % Left singular functions evaluated at x

% SysID
temp=D*GrInv;
f = @(x) temp*squeeze(pagemtimes(Kr.K(x,X),w)); % Vector field
end