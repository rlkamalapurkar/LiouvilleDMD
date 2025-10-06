% The KernelRKHS class implements the following kernels:
%
% 'Exponential with parameter mu: exp(x.'*y/mu)', 
% 'Gaussian with parameter mu: exp(norm(x-y)^2/mu)',
% 'Wendland with parameter [support,d,k]: C2k smooth in d dimensions',
% 'Polynomial with parameters [mu,d,b]: (b + x.'*y/mu)^d', and
% 'Linear with parameter mu = Polynomial with parameters [mu,1,0]'
%
% The kernels functions are written to accept column vector, matrix, 
% or 3D array inputs. The two inputs X and Y have to be the same size in 
% dimensions 1 and 2. If they have different number of pages, one of them 
% has to be a single page. 
% 
% Let the kernel function be denoted by k. The class method obj.K returns
% a u x u x p matrix M, where u = size(X,2), and 
% p = max(size(X,3), size(Y,3)), with 
% 
% M(i,j,k) = k(X(:,i,k),Y(:,j,k)) if if X and Y are same size,
% M(i,j,k) = k(X(:,i,k),Y(:,j,1)) if size(Y,3) = 1, and
% M(i,j,k) = k(X(:,i,1),Y(:,j,k)) if size(X,3) = 1.
%
% The class also includes a method obj.G2K that calculates the inner 
% product of a given vector with the gradient of the kernel function obj.K,
% with respect to the second argument. This function is compatible
% with vector, matrix, or 3D array inputs for the first two arguments. 
% ***The third argument has to be a column vector.***
% 
% Let the gradient $\frac{\partial k(x,y)}{\partial y}$ be denoted by Gk.
% obj.G2K(X,Y,z) returns a matrix N, the same size as M above, with
% 
% N(i,j,k) = z^T * Gk(X(:,i,k),Y(:,j,k)) if X and Y are same size,
% N(i,j,k) = z^T * k(X(:,i,k),Y(:,j,1)) if size(Y,3) = 1, and
% N(i,j,k) = z^T * k(X(:,i,1),Y(:,j,k)) if size(X,3) = 1.
%
% © Rushikesh Kamalapurkar
%
classdef KernelRKHS
    properties
        p; % kernel parameter vector
        type; % kernel type
        coeffs; % Wendland polynomial coefficients
        diffCoeffs; % Wendland gradient polynomial coefficients
    end
    methods
        function obj = KernelRKHS(type,p)
            % Processing arguments and setting defaults
            if nargin == 2
                obj.type = type;
                obj.p = p;
            elseif nargin == 0
                obj.type = 'Gaussian'; % default
                obj.p = 1; % default
                warning('Setting kernel type to Gaussian and kernel parameter to 1');
            else
                error('Two (or zero) input arguments needed');
            end
            if isequal(obj.type,'Polynomial')
                if isscalar(p)
                    obj.p(1) = p;
                    obj.p(2) = 1;
                    obj.p(3) = 0;
                    warning('Selected linear kernel with parameter mu: k(x,y) = x.''*y/mu')
                elseif numel(p)==3
                    obj.p(1) = p(1);
                    obj.p(2) = p(2);
                    obj.p(3) = p(3);
                else
                    error('The polynomial kernel requires three parameters [mu,d,b], k(x,y) = (b + x.''*y/mu)^d')
                end
            else
                obj.p(2) = 1;
                obj.p(3) = 0;
            end
            if isequal(obj.type,'Linear')
                obj.p(1) = p;
                obj.p(2) = 1;
                obj.p(3) = 0;
            end
            if isequal(obj.type,'Wendland')
                if isscalar(p)
                    obj.p(1) = p;
                    obj.p(2) = 2;
                    obj.p(3) = 2;
                    warning(['Selected Wendland kernel with support [0,' num2str(p) '], with C4 smoothness in 2 dimensions'])
                elseif numel(p)==3
                    obj.p(1) = p(1);
                    obj.p(2) = p(2);
                    obj.p(3) = p(3);
                else
                    error('The Wendland kernel requires three parameters [support,dimension,smoothness]')
                end
                l = floor(obj.p(2)/2) + obj.p(3) + 1;
                [obj.coeffs,obj.diffCoeffs] = obj.generatePoly(l,obj.p(3));
            end
        end

        function y = K(obj,X,Y)
            % Kernel function
            if isequal(obj.type,'Gaussian')
                y = exp(-1/obj.p(1)*(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none')));
            elseif isequal(obj.type,'Wendland')
                X = X./obj.p(1);
                Y = Y./obj.p(1);
                r = sqrt(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'));
                y = (r<=1).*polyval(obj.coeffs,r);
            elseif isequal(obj.type,'Exponential')
                y = exp(1/obj.p(1)*pagemtimes(X,'transpose',Y,'none'));
            elseif isequal(obj.type,'Linear') || isequal(obj.type,'Polynomial')
                y = (obj.p(3) + 1/obj.p(1)*pagemtimes(X,'transpose',Y,'none')).^obj.p(2);
            else
                error(['Kernel type ' obj.type ' not implemented']);
            end
        end

        function y = G2K(obj,X,Y,Z)
            % A function that calculates the product of a given vector and
            % the gradient of K with respect to the second argument.
            if isequal(obj.type,'Gaussian')
                y = (2/obj.p(1))*(pagemtimes(X,'transpose',Z,'none') - pagemtimes(Z,'transpose',Y,'none')).*obj.K(X,Y);
            elseif isequal(obj.type,'Wendland')
                if obj.p(3) == 0
                    error('Wendland kernel with smoothness 0 is not differentiable')
                end
                X = X./obj.p(1);
                Y = Y./obj.p(1);
                r = sqrt(pagetranspose(sum(X.^2,1)) + sum(Y.^2,1) - 2*pagemtimes(X,'transpose',Y,'none'));
                y = ((r<=1).*polyval(obj.diffCoeffs,r)).*(pagemtimes(Y - X,'transpose',Z,'none'));
            elseif isequal(obj.type,'Exponential')
                y = 1/obj.p(1)*pagemtimes(X,'transpose',Z,'none').*obj.K(X,Y);
            elseif isequal(obj.type,'Linear') || isequal(obj.type,'Polynomial')
                if obj.p(2) == 1
                    y = 1/obj.p(1)*pagemtimes(X,'transpose',Z,'none').*ones(size(X,2),size(Y,2),max(size(X,3),size(Y,3)));
                else
                    y = obj.p(2)/obj.p(1)*pagemtimes(X,'transpose',Z,'none').*(obj.p(3) + 1/obj.p(1)*pagemtimes(X,'transpose',Y,'none')).^(obj.p(2)-1);
                end
            else
                error(['Kernel type ' obj.type ' not implemented']);
            end
        end

        function y = OKGram(obj,X,wx,Y,wy)
            y=zeros(size(X,3),size(Y,3));
            for i=1:size(Y,3)
                y(:,i) = squeeze(pagemtimes(pagemtimes(wx(:,1,i).',obj.K(X(:,:,i),Y)),wy));
            end
        end

        function y = OK(obj,x,X,w)
            y=zeros(size(X,3),size(x,2));
            for i=1:size(x,2)
                y(:,i) = squeeze(pagemtimes(obj.K(x,X),w));
            end
        end

        function [y,dy] = generatePoly(obj,l,k)
            % Generate the coefficients of the Wendland kernel polynomial
            % of degree l and smoothness k
            d = NaN*ones(l+2*k+1,k+1);
            dj0 = arrayfun(@(j) (-1)^j*nchoosek(l,j),(0:l).');
            d(1:l+1,1) = dj0;
            if k>0
                d1splus1 = zeros(1,k);
                d(2,2:end) = d1splus1;
                for s = 0:k-1
                    d0splus1 = 0;
                    for j = 0:l+2*s
                        djs = d(j+1,s+1);
                        d0splus1 = d0splus1 + djs/(j+2);
                    end
                    d(1,s+2) = d0splus1;
                    for j = 2:l+2*s+2
                        djminus2s = d(j-1,s+1);
                        djsplus1 = -djminus2s/j;
                        d(j+1,s+2) = djsplus1;
                    end
                end
            end
            y = d(:,end).';
            y = flip(y/y(1));
            if k > 0
                dy = polyder(y);
                dy = dy(1:end-1);
            else
                dy = [];
            end
        end
    end
end