% This function performs dynamic mode decomposition of a second order 
% dynamical system from sampled trajectories. For details, see
% 
% https://arxiv.org/abs/2101.02646
%
% [Z,L,ef,r,f] = SecondOrderLiouvilleDMD(K,G2K,W,t) OR
% [Z,L,ef,r,f] = SecondOrderLiouvilleDMD(K,G2K,W,t,WDot0) OR
% [Z,L,ef,r,f] = SecondOrderLiouvilleDMD(K,G2K,W,t,WDot0,l)
%
% Inputs:
%    1) K: An object of the class Kernel.
%          K.K needs to be the kernel function, compatible with 3D arrays. 
%          K.G2K needs to be a function that calculates the product of a 
%          given vector and the gradient of K with respect to the second
%          argument.
%          Example 1, Gaussian:
%             K.K = @(X,Y) exp(-1/mu*(pagetranspose(sum(X.^2,1)) + ...
%             sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'))); 
%             K.G2K = @(X,Y,Z) pagemtimes(2*(X - Y)/mu,...
%              'transpose', Z, 'none').*K(X,Y);
%          Example 2, exponential dot product:
%             K.K = @(X,Y) exp(1/mu*pagemtimes(X,'transpose',Y,'none'));
%             K.G2K = @(X,Y,Z) 1/mu * ...
%              pagemtimes(X,'transpose',Z,'none').*K(X,Y);
%          Example 2, linear dot product:
%             K.K = @(X,Y) 1/mu*pagemtimes(X,'transpose',Y,'none');
%             K.G2K = @(X,Y,Z) 1/mu*pagemtimes(X,'transpose',Z,'none');
%    2) W: A dataset of trajectories (3D array)
%          First dimension: State 
%          Second dimension: Time (size = length of longest trajectory)
%          Third dimension: Trajectory number
%
% *Trajectories can be of different lengths and irregularly sampled.*
% *the number of samples in each trajectory needs to be odd.*
% *Shorter trajectories need to be padded with zeros.*
%
%    3) t: A matrix of sample times
%          First dimension: Time (size = length of longest trajectory)
%          Second dimension: Trajectory number
%
% *Sample times for shorter trajectories need to be padded with NaNs.*
%
%    5) WDot0: (Optional argument) Initial time derivatives
%          First dimension: State 
%          Second dimension: Trajectory number
%    6) l: (Optional argument) Regularization coefficient
%
% Outputs:
%          1) Z: Liouville modes
%          2) L: Liouville eigenvalues
%          3) ef: Liouville eigenfunctions
%          4) r: Reconstruction function
%          5) f: Vector Field
%
% © Rushikesh Kamalapurkar, Ben Russo, and Joel Rosenfeld
%
function [Z,L,ef,r,f] = SecondOrderLiouvilleDMD(K,W,t,varargin)
if nargin == 3
    if size(W,1)==1
        WDiff = (squeeze(W(:,2,:) - W(:,1,:))).';
    else
        WDiff = squeeze(W(:,2,:) - W(:,1,:));
    end
    WDot0 = WDiff./(t(2,:) - t(1,:));
    l = 0;
elseif nargin == 4
    WDot0 = varargin{1};
    l = 0;
elseif nargin == 5
    WDot0 = varargin{1};
    if isempty(WDot0)
        if size(W,1)==1
            WDiff = (squeeze(W(:,2,:) - W(:,1,:))).';
        else
            WDiff = squeeze(W(:,2,:) - W(:,1,:));
        end
        WDot0 = WDiff./(t(2,:) - t(1,:));
    end
    l = varargin{2};
elseif nargin > 5
    error("Too many input arguments");
end
    
N = size(W,3); % Total number of trajectories

% Store trajectory lengths for interaction matrix calculation
Lengths = size(t,1) - sum(isnan(t));

% Simpsons rule weights
S = reshape(genSimpsonsRuleWeights(t,1),size(t,1),1,size(t,2));
t(isnan(t))=0;
T=zeros(1,1,size(t,2));
for i=1:size(t,2)
    if Lengths(i) == 1
        T(1,1,i) = t(1,i);
    else
        T(1,1,i) = t(find(t(:,i),1,'last'),i);
    end
end
t = reshape(t,size(t,1),1,size(t,2));
% Gram matrix and interaction matrix
G=zeros(N);
I=zeros(N);
ST=S.*(T-t);
for j=1:N
    G(:,j) = squeeze(pagemtimes(ST,'transpose',pagemtimes(K.K(W,W(:,:,j)),ST(:,1,j)),'none'));
    I(:,j) = squeeze(pagemtimes(ST,'transpose',K.K(W,W(:,Lengths(j),j)) - K.K(W,W(:,1,j)) - T(1,1,j).*K.G2K(W,W(:,1,j),WDot0(:,j)),'none'));
end

% DMD
G = G + l*eye(size(G));
[V,D] = eig(G\I.'); % Eigendecomposition of finite rank representation
C = V./diag(sqrt(V'*G*V)).'; % Liouville eigenfunction coefficients
IntMat = squeeze(pagemtimes(W,ST)); % Integrals of trajectories
if size(W,1)==1
    IntMat = IntMat.';
end
Z = IntMat/(C.'*G); % Liouville modes
L = diag(D); % Liouville eigenvalues

% Reconstruction
% Occupation kernels evaluated at x0: 
%   squeeze(pagemtimes(ST,'transpose',K.K(W,x0),'none'))
% Eigenfunctions evaluated at x0:
ef = @(x) C.'*squeeze(pagemtimes(ST,'transpose',K.K(W,x),'none'));
% Time derivative of eigenfunctions at t = 0:
%   C.'*squeeze(pagemtimes(ST,'transpose',K.G2K(W,x0,xDot0),'none'))
% Reconstruction function:
r = @(t,x0,xDot0) real(Z*(...
    0.5*(C.'*squeeze(pagemtimes(ST,'transpose',K.K(W,x0),'none'))...
    + (C.'*squeeze(pagemtimes(ST,'transpose',K.G2K(W,x0,xDot0),'none')))./sqrt(L)).*exp(sqrt(L)*t) +...
    0.5*(C.'*squeeze(pagemtimes(ST,'transpose',K.K(W,x0),'none'))...
    - (C.'*squeeze(pagemtimes(ST,'transpose',K.G2K(W,x0,xDot0),'none')))./sqrt(L)).*exp(-sqrt(L)*t)));
% Vector field
f = @(x) real(Z*D*C.'*squeeze(pagemtimes(ST,'transpose',K.K(W,x),'none')));
end