% This function performs occupation kernel regression of a control affine
% dynamical system
%
% x_dot = f(x) + g(x)u
%
% from sampled state and control trajectories. Higher-order (Order > 1)
% regression, where the occupation kernel integrand is weighted by
% (T-t)^(Order-1)/(Order-1)! and the initial derivatives are estimated
% using finite differences, is also supported.
%
% fg = ControlOccupationKernelRegression(K,X,U,t) OR
% fg = ControlOccupationKernelRegression(K,X,U,t,RegTol=l) OR
% fg = ControlOccupationKernelRegression(K,X,U,t,PinvTol=l) OR
% fg = ControlOccupationKernelRegression(K,X,U,t,Order=r)
%
% Inputs:
%    1) K:       vvRKHS kernel with (control dimension + 1) parameters
%    2) X:       A dataset of state trajectories
%                First dimension: State
%                Second dimension: Time (size = length of longest trajectory)
%                Third dimension: Trajectory number
%    3) U:       A dataset of control trajectories
%                First dimension: Control
%                Second dimension: Time (size = length of longest trajectory)
%                Third dimension: Trajectory number
%    4) t:       A matrix of sample times, padded with NaN
%                First dimension: Time (size = length of longest trajectory)
%                Second dimension: Trajectory number
%                Trajectory lengths need to be odd (Simpson's rule). For
%                Order > 1, the first Order samples of each trajectory need
%                to be uniformly spaced.
%    5) RegTol:  (Optional argument, default=0) Regularization coefficient
%    6) PinvTol: (Optional argument, default=0) Pseudoinverse coefficient
%    7) Order:   (Optional argument, default=1) Order of the regression
%
% If RegTol and PinvTol are both nonzero or both zero, or if PinvTol is
% supplied but RegTol is not, then pseudoinverse is used to invert the Gram
% matrix. If RegTol is supplied and PinvTol is not, then regularization
% is used to invert the Gram matrix.
%
% Outputs:
%    1) fg:      function that maps x to [f(x) g(x)]
%
% Based on code by Joel A. Rosenfeld (2021).
%
function fg = ControlOccupationKernelRegression(K,X,U,t,NameValueArgs)

arguments
    K
    X double
    U double
    t double
    NameValueArgs.RegTol (1,1) {mustBeNumeric} = NaN
    NameValueArgs.PinvTol (1,1) {mustBeNumeric} = NaN
    NameValueArgs.Order (1,1) {mustBeInteger,mustBePositive} = 1
end
if ~isnan(NameValueArgs.RegTol) && ~isnan(NameValueArgs.PinvTol)
    warning('RegTol and PinvTol are both nonzero, defaulting to pseudoinverse. Call with RegTol ~= 0 and PinvTol = 0 to use regularization.');
    NameValueArgs.RegTol = NaN;
end
r = NameValueArgs.Order;

M = size(X,3); % Number of trajectories
m = size(U,1); % Control dimension
n = size(X,1); % State dimension
maxN = size(X,2); % Number of samples in longest trajectory
N = size(t,1)-sum(isnan(t),1); % Trajectory lengths
if size(U,3) ~= M || size(U,2) < maxN
    error('Control and state trajectories are incompatible.')
end

% Simpson's rule weights, multiplied by the Taylor remainder kernel
w = genSimpsonsRuleWeights(t,1);
T = t(sub2ind(size(t),N,1:M)) - t(1,:); % Trajectory durations
tau = (t(sub2ind(size(t),N,1:M)) - t).^(r-1)/factorial(r-1);
tau(isnan(tau)) = 0;
w = reshape(w.*tau,size(t,1),1,M);
w(isnan(w)) = 0;

% Concatenate control array with 1 in the first dimension
U = cat(1,ones(1,maxN,M),U(:,(1:maxN),:));

% Make 3D control array into 4D array for Gram matrix calculations
U = reshape(permute(U,[2,3,1]),maxN,1,M,m+1);

% Control OCC Gram matrix - M x M
Gb = zeros(M);
% Differences of endpoints and Taylor polynomials of the initial conditions
D = zeros(n,M);
for i=1:M
    Xi = X(:,1:N(i),i);
    iv = Xi(:,1);
    if r > 1
        h = t(2,i) - t(1,i);
        for j=1:r-1
            dj = diff(Xi(:,1:j+1),j,2)/h^j;
            iv = iv + dj*T(i)^j/factorial(j);
        end
    end
    D(:,i) = Xi(:,end) - iv;
    Gb(:,i) = squeeze(pagemtimes(pagemtimes(w(:,1,i).',sum(K.K(X(:,:,i),X).*pagemtimes(U(:,:,i,:),'none',U,'transpose'),4)),w));
end

% Weights
if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
    GbInv = pinv(Gb);
elseif ~isnan(NameValueArgs.PinvTol)
    GbInv = pinv(Gb,NameValueArgs.PinvTol);
else
    GbInv = pinv(Gb+NameValueArgs.RegTol*eye(size(Gb)));
end
temp = D*GbInv.';

% Vector field
fg = @(x) temp*squeeze(pagemtimes(K.K(x,X).*permute(U,[2,1,3,4]),w));
end
