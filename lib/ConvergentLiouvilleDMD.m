% This function performs provably convergent dynamic mode decomposition of 
% a dynamical system from sampled trajectories. For details, see
%
% https://arxiv.org/abs/2106.02639
%
% [Z,S,lsf,rsf,f] = ConvergentLiouvilleEigenDMD(Kd,Kr,X,T) OR
% [Z,S,lsf,rsf,f] = ConvergentLiouvilleEigenDMD(Kd,Kr,X,T,l)
%
% Inputs:
%  1,2) Kd,Kr: Objects of the class 'Kernel' for the domain and the range,
%              Kr.K is the kernel function
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
%    2) S: Singular values (number of modes x number of modes)
%    3) lsf: Left singular functions
%    4) rsf: Right singular functions
%    4) f: Approximation of the vector field
%
% © Rushikesh Kamalapurkar
%
function [Z,S,lsf,rsf,f] = ConvergentLiouvilleDMD(Kd,Kr,X,t,varargin)
% Processing optional arguments and setting defaults
if nargin == 4
    l = 0; % default
elseif nargin == 5
    l = varargin{1};
elseif nargin > 5
    error("Too many input arguments.")
end

N = size(X,3); % Total number of trajectories
n = size(X,1); % State Dimension

% Store trajectory lengths for interaction matrix calculation
Lengths = size(t,1)-sum(isnan(t));

% Simpsons rule weights
w = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2));

% Gram matrix and interaction matrix
GT=zeros(N);
D = zeros(n,N);
for i=1:N
    GT(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',Kr.K(X(:,:,i),X)),w));
    D(:,i) = X(:,Lengths(i),i) - X(:,1,i);
end
% DMD
GT = GT + l*eye(size(GT)); % Regularization
[VT,S,V] = svd(inv(GT)); % SVD of GT
Z = D*V; % Liouville modes

% Occupation kernels evaluated at x: squeeze(pagemtimes(K(x,W),S))
% Right singular functions evaluated at x:
rsf = @(x) VT.'*squeeze(pagemtimes(Kr.K(x,X),w));
% Left singular functions evaluated at x:
lsf = @(x) V.'*(arrayfun(@(l) Kd.K(x,X(:,Lengths(l),l)),(1:M).') ...
    - arrayfun(@(l) Kd.K(x,X(:,1,l)),(1:M).'));

% Vector field:
temp=D/GT.';
f = @(x) real(temp*squeeze(pagemtimes(Kr.K(x,X),w)));
end