% The KernelVectorRKHS class implements vector-valued versions of
% the following kernels:
%
% 'Exponential', 
% 'Gaussian'
% 'Linear'
%
% For now, only diagonal kernel operators of the form
% 
% K(x,y) = diag(k_1(x,y;p_1),...,k_m(x,y;p_m))
%
% are supported where the kernel functions k_1,...,k_m on the diagonal vary
% only in their parameter. The parameter values are supplied by the user as
% a column vector.
%
% The kernel functions accept column vector, matrix, or 3D array inputs. 
% The two inputs X and Y have to be the same size in dimensions 1 and 2. 
% If they have different number of pages, one of them has to be a single
% page. 
% 
% The class method obj.K returns a u x u x p x m array M, where
% u = size(X,2), m = size(parameter,1) and p = max(size(X,3), size(Y,3)) 
% with 
% 
% M(i,j,k,l) = k_l(X(:,i,k),Y(:,j,k)) if X and Y are same size,
% M(i,j,k,l) = k_l(X(:,i,k),Y(:,j,1)) if size(Y,3) = 1, and
% M(i,j,k,l) = k_l(X(:,i,1),Y(:,j,k)) if size(X,3) = 1.
%
% © Rushikesh Kamalapurkar
%
classdef KernelvvRKHS
    properties
        p; % kernel parameters in a column vector
        type; % kernel type
        coeffs; % Wendland polynomial coefficients
    end
    methods
        function obj = KernelvvRKHS(type,p)
            % Processing arguments and setting defaults
            if nargin == 2
                obj.type = type;
                if p > 0
                    obj.p = p;
                else
                    error('The kernel parameter needs to be positive')
                end
            elseif nargin == 0
                obj.type = 'Gaussian'; % default
                obj.p = 1; % default
                warning('Setting kernel dimension to 1, kernel type to Gaussian, and kernel parameter to 1');
            else
                error('Two (or zero) input arguments needed');
            end
        end

        function y = K(obj,X,Y)
            % Kernel operator
            if isequal(obj.type,'Gaussian')
                y = exp(pagemtimes(reshape(-1./obj.p,1,1,1,numel(obj.p)),repmat(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'),1,1,1,numel(obj.p))));
            elseif isequal(obj.type,'Wendland34')
                X = X./obj.p;
                Y = Y./obj.p;
                r = sqrt(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'));
                y = repmat(max(0,(1-r).^6).*(35*r.^2+18*r+3),1,1,1,numel(obj.p));
            elseif isequal(obj.type,'Wendland54')
                X = X./obj.p;
                Y = Y./obj.p;
                r = sqrt(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'));
                y = repmat(max(0,(1-r).^7).*(16*r.^2+7.*r+1),1,1,1,numel(obj.p));
            elseif isequal(obj.type,'Exponential')
                y = exp(pagemtimes(reshape(1./obj.p,1,1,1,numel(obj.p)),repmat(pagemtimes(X,'transpose',Y,'none'),1,1,1,numel(obj.p))));
            elseif isequal(obj.type,'Linear')
                y = pagemtimes(reshape(1./obj.p,1,1,1,numel(obj.p)),repmat(pagemtimes(X,'transpose',Y,'none'),1,1,1,numel(obj.p)));
            else
                error(['Kernel type' obj.type 'not implemented']);
            end
        end
    end
end
                