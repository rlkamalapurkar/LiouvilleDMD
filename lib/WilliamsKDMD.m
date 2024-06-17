% This function computes the modes, eigenvalues, and eigenfunction
% coefficients of a finite-rank representation of the Koopman operator
% for the vector field that relates the inputs X to the outputs Y. This is 
% an efficient implementation of the kernel DMD method in
%
% https://doi.org/10.3934/jcd.2015005
%
% [Z,L,ef,cr,dr,fc] = WilliamsKDMD(X,Y,K,deltaT) OR
% [Z,L,ef,cr,dr,fc] = WilliamsKDMD(X,Y,K,deltaT,eigTol)
%
% Inputs:
%       X: Input data,
%           First dimension: state variables
%           Second dimension: snapshots
%       Y: Output data, Y = f(X)
%       K: Object of the class 'Kernel', where K.K is the kernel function
%          Example 1, Gaussian:
%             K.K = @(X,Y) exp(-1/mu*(pagetranspose(sum(X.^2,1)) + ...
%             sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'))); 
%          Example 2, exponential dot product:
%             K.K = @(X,Y) exp(1/mu*pagemtimes(X,'transpose',Y,'none'));
%       deltaT: Sample time
%       eigTol: (Optional argument, default=0) Eigenvalues that are smaller
%               in absolute value than eigTol are removed along with the
%               corresponding eigenvectors.
% Outputs:
%       Z: Koopman modes
%       L: Koopman eigenvalues
%       ef: Koopman eigenfunctions
%       cr: Continuous Koopman reconstruction function, cr(t,x0)
%           reconstructs the system state at time t starting from x0
%       dr: Discrete Koopman reconstruction function, dr(k,x0) reconstructs
%           the k-th snapshot starting from x0. With k = 1, we get the
%           vector field $ f_{d} $ of the estimated discrete time system as
%           $ x_{k+1} = fd(x_{k}) = dr(1,x_{k}) $
%       fc: The vector field of the estimated continuous time system 
%           $ \dot{x} = fc(x) $
%
% © Rushikesh Kamalapurkar
%
function [Z,L,ef,cr,dr,fc] = WilliamsKDMD(X,Y,K,deltaT,varargin)
    % Processing optional arguments and setting defaults 
    if nargin == 4
        eigTol = 0; % default
    elseif nargin == 5
        eigTol = varargin{1};
    elseif nargin > 5
        error("Too many input arguments.")
    end
    % Gram matrix and interaction matrix
    GHat = K.K(X,X);
    AHat = K.K(X,Y).';
    % DMD using the procedure in Williams et al., 2015
    [Q,SigmaSquared] = eig(GHat,'vector');
    if eigTol~=0 % remove "zero" eigenvalues
        Q(:,abs(SigmaSquared)<=eigTol)=[];
        SigmaSquared(abs(SigmaSquared)<=eigTol)=[];
    end
    Sigma = diag(sqrt(SigmaSquared));
    KHat = (pinv(Sigma)*Q')*AHat*(Q*pinv(Sigma));
    [VHat,L] = eig(KHat,'vector');
    if eigTol~=0 % remove "zero" eigenvalues
        VHat(:,abs(L)<=eigTol)=[];
        L(abs(L)<=eigTol)=[];
    end
    contL = log(L)./deltaT; % Continuous-time eigenvalues
    ef = @(x) (K.K(x,X)*(Q*pinv(Sigma)*VHat)).'; % Koopman eigenfunctions
    Z = (pinv(VHat)*pinv(Sigma)*Q'*X').'; % Koopman modes
    % Discrete Koopman reconstruction function
    dr = @(k,x0) real(Z*(ef(x0).*L.^k));
    % Continuous Koopman reconstruction function
    cr = @(t,x0) real(Z*(ef(x0).*exp(contL*t)));
    % Continuous Koopman vectorifield
    fc = @(x0) real(Z*(ef(x0).*contL));
end
