% This function performs dynamic mode decomposition of a controlled 
% dynamical system from snapshots (x,u,y) of the form
% 
% y = f(x) + g(x)u
%
% fg = ControlKoopmanDMDOpenLoop(KT,K,X,U,Y,PinvTol=l) OR
% fg = ControlKoopmanDMDOpenLoop(KT,K,X,U,Y,Regtol=l) OR
% fg = ControlKoopmanDMDOpenLoop(KT,K,X,U,Y)
%
% Inputs:
%    1) KT:      svRKHS kernel 
%    2) K:       vvRKHS kernel 
%    3) X:       Input data
%                First dimension: State
%                Second dimension: Snapshot
%    4) U:       Control input data
%                First dimension: Control
%                Second dimension: Snapshot
%    5) Y:       Output data, Y(:,k) = f(X(:,k)) + g(x)U(:,k)
%                First dimension: State
%                Second dimension: Snapshot
%    6) RegTol:  (Optional argument, default=0) Regularization coefficient
%    7) PinvTol: (Optional argument, default=0) Pseudoinverse coefficient
%
% If RegTol and PinvTol are both nonzero or both zero, or if PinvTol is  
% supplied but RegTol is not, then pseudoinverse is used to invert Gram 
% matrices. If RegTol is supplied and PinvTol is not, then regularization 
% is used to invert Gram matrices.
%
% Outputs:
%    1) fg:      function that maps x to [f(x) g(x)]
%
% © Rushikesh Kamalapurkar and Moad Abudia, 2025
%
function fg = ControlKoopmanDMDOpenLoop(KT,K,X,U,Y,NameValueArgs)

arguments
    KT
    K
    X double
    U double
    Y double
    NameValueArgs.RegTol (1,1) {mustBeNumeric} = NaN
    NameValueArgs.PinvTol (1,1) {mustBeNumeric} = NaN
end
if ~isnan(NameValueArgs.RegTol) && ~isnan(NameValueArgs.PinvTol)
    warning('RegTol and PinvTol are both nonzero, defaulting to pseudoinverse. Call with RegTol ~= 0 and PinvTol = 0 to use regularization.');
    NameValueArgs.RegTol = NaN;
end

% Values of the controller
U = permute([ones(1,size(U,2));U],[2 1 3]);

% vvRKHS Gram matrix and interaction matrix - M x M
temp = squeeze(K.K(X,X));
G = temp(:,:,1).*(U*U.');

% svRKHS Gram matrix and interaction matrix - M x M
GT = KT.K(X,X);
IT = KT.K(X,Y);

% DMD
if (isnan(NameValueArgs.PinvTol) && isnan(NameValueArgs.RegTol)) || (~isnan(NameValueArgs.PinvTol) && NameValueArgs.PinvTol == 0)
    GTInv = pinv(GT);
    GInv = pinv(G);
    % disp('Inverting Gram matrices using pseudoinverse.')
elseif ~isnan(NameValueArgs.PinvTol)
    GTInv = pinv(GT,NameValueArgs.PinvTol);
    GInv = pinv(G,NameValueArgs.PinvTol);
    % disp('Inverting Gram matrices using truncated pseudoinverse.')
elseif ~isnan(NameValueArgs.RegTol)
    GTInv=pinv(GT+NameValueArgs.RegTol*eye(size(GT))); % Regularization
    GInv=pinv(G+NameValueArgs.RegTol*eye(size(G))); % Regularization
    % disp('Inverting Gram matrices using regularization.')
end
FRR=GInv*IT.';
[W,S,V] = svd(FRR*GTInv);
if ~isnan(NameValueArgs.PinvTol)
    S(S<NameValueArgs.PinvTol) = 0;
end

% Vector fields
fg = @(x) X*V*S*W.'*(squeeze(K.K(x,X)).*U);
end