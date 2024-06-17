% This function computes the modes, eigenvalues, and eigenfunction
% coefficients of a finite-rank representation of the Koopman operator
% for the vector field that relates the inputs X to the outputs Y. This is 
% an efficient implementation of the kernel DMD method in
%
% https://arxiv.org/abs/2106.00106
%
% [Z,L,ef,cr,dr,fc] = KoopmanDMD(X,Y,K,deltaT) OR
% [Z,L,ef,cr,dr,fc] = KoopmanDMD(X,Y,K,deltaT,RegTol=l)
% [Z,L,ef,cr,dr,fc] = KoopmanDMD(X,Y,K,deltaT,PinvTol=l)
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
%          Example 3, linear dot product
%             K.K = @(X,Y) 1/mu*pagemtimes(X,'transpose',Y,'none');
%       deltaT: Sample time
%       l: (Optional argument, default=0) Regularization coefficient or
%          pseudoinverse truncation tolerance. By default, the Gram matrix
%          is inverted using MATLAB's pinv function. If PinvTol is
%          supplied, it is passed to pinv as tolerance. If RegTol is set
%          instead, then the Gram matrix G is regularized as G = G + lI
%          before it is inverted. Eigenvalues that are smaller
%          in absolute value than l are also removed along with the
%          corresponding eigenvectors.
% Outputs:
%       Z:  Koopman modes
%       L:  Koopman eigenvalues
%       ef: Koopman eigenfunctions
%       cr: Continuous Koopman reconstruction function, cr(t,x0)
%           reconstructs the system state at time t starting from x0
%       dr: Discrete Koopman reconstruction function, dr(k,x0) reconstructs
%           the k-th snapshot starting from x0. With k = 1, we get the
%           vector field $ f_{d} $ of the estimated discrete time system as
%           $ x_{k+1} = fd(x_{k}) = dr(1,x_{k}) $
%       fc: The vector field of the estimated continuous time system 
%           $ \dot{x} = fc(x) $. This is relevant only if the data were
%           sampled from a continuous time system.
%
% © Rushikesh Kamalapurkar
%
function [Z,L,ef,cr,dr,fc] = KoopmanDMD(X,Y,K,deltaT,NameValueArgs)
    % Process optional arguments and set defaults
    arguments
        X double
        Y double
        K
        deltaT double
        NameValueArgs.RegTol (1,1) {mustBeNumeric} = NaN
        NameValueArgs.PinvTol (1,1) {mustBeNumeric} = NaN
    end
    if ~isnan(NameValueArgs.RegTol) && ~isnan(NameValueArgs.PinvTol)
        warning('RegTol and PinvTol are both nonzero, defaulting to pseudoinverse. Call with RegTol ~= 0 and PinvTol = 0 to use regularization.');
        NameValueArgs.RegTol = NaN;
    end
    % Gram matrix and interaction matrix
    G = K.K(X,X);
    I = K.K(X,Y);
    % DMD (see https://arxiv.org/abs/2106.00106 for details)
    if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
        FRR = pinv(G)*I.'; % pseudoinverse
        eigTol = 0;
    elseif ~isnan(NameValueArgs.PinvTol)
        FRR = pinv(G,NameValueArgs.PinvTol)*I.';
        eigTol=NameValueArgs.PinvTol; % truncated pseudoinverse
    elseif ~isnan(NameValueArgs.RegTol)
        FRR = (G + NameValueArgs.RegTol*eye(size(G)))\I.';
        eigTol=NameValueArgs.RegTol; % regularization
    end
    [V,L] = eig(FRR,'vector'); % Eigendecomposition of finite rank representation
    if eigTol~=0 % remove "zero" eigenvalues
        V(:,abs(L)<=eigTol)=[];
        L(abs(L)<=eigTol)=[];
    end
    contL = log(L)./deltaT; % Continuous-time eigenvalues
    C = V./diag(sqrt(V'*G*V)).'; % Koopman eigenfunction coefficients
    if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
        Z = X*pinv(C.'*G); % pseudoinverse
    elseif ~isnan(NameValueArgs.PinvTol)
        Z = X*pinv(C.'*G,NameValueArgs.PinvTol); % truncated pseudoinverse
    elseif ~isnan(NameValueArgs.RegTol)
        Z = X/(C.'*(G + NameValueArgs.RegTol*eye(size(G)))); % regularization
    end
    ef = @(x0) (K.K(x0,X)*C).';% Koopman eigenfunctions
    % Discrete Koopman reconstruction function (or vector field if k = 1)
    dr = @(k,x0) real(Z*(ef(x0).*L.^k));
    % Continuous Koopman reconstruction function
    cr = @(t,x0) real(Z*(ef(x0).*exp(contL*t)));
    % Continuous Koopman vector field
    fc = @(x0) real(Z*(ef(x0).*contL));
end
