function  S = NNLS(A, X, opts)


% Author: Minchao Ye

% error(nargchk(3, 4, nargin, 'struct'));


maxit = 3000;
tol = 1e-6;
verbose = false;

delta=50;

if exist('opts', 'var')
    if isfield(opts, 'maxit')
        maxit = opts.maxit;
    end
    if  isfield(opts, 'tol')
        tol = opts.tol;
    end
    if  isfield(opts, 'verbose')
        verbose = opts.verbose;
    end
end


if verbose
    tStart = tic;
end


if isa(X, 'double')
    eps_val = 1e-16;
elseif isa(X, 'single')
    eps_val = single(1e-8);
else
    error('Invalid input data type.');
end

class_type = class(X);

% [n, m] = size(X);

r=size(A,2);


A = [A;delta*ones(1,size(A,2),class_type)];
X = [X;delta*ones(1,size(X,2),class_type)];

% S = rand(r, m, class_type);
% S = max(S, eps_val);

S=A\X;

C = inf;

S_row_max = max(S, [], 2);
[max_S, max_S_row_idx] = max(S_row_max);


for itrcount = 1:maxit
    U = A'*X;
    V = A'*A;
    idx = randperm(r);
    for j_count = 1:r
        j = idx(j_count);
        W = S(j,:)+(U(j,:)-V(j,:)*S)/V(j,j);
        W = max(W, eps_val*max_S);
        S(j,:) = W;
        
        S_row_max(j) = max(W);
        if S_row_max(j) >= max_S
            max_S = S_row_max(j);
            max_S_row_idx = j;
        elseif j == max_S_row_idx
            [max_S, max_S_row_idx] = max(S_row_max);
        end
    end
       
    C_old = C;
    C = 1/2*norm(X(1:(end-1),:)-A(1:(end-1),:)*S, 'fro')^2;
    if verbose
        fprintf('Iteration %d.\tObjective function value C=%f\n', itrcount, C);
        toc(tStart);
    end
    if abs(C_old-C) < tol*C
        break;
    end
end


end

