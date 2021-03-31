function [A, S] = NMF(X, r, opts)


% Author: Minchao Ye

% error(nargchk(3, 4, nargin, 'struct'));


maxit = 3000;
tol = 1e-5;
verbose = true;

delta=20;

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

[n, m] = size(X);

A = 2*mean(X(:))*rand(n, r, class_type);
A = max(A, eps_val);
A = [A;delta*ones(1,size(A,2),class_type)];
X = [X;delta*ones(1,size(X,2),class_type)];

S = rand(r, m, class_type);
S = max(S, eps_val);

C = inf;

S_row_max = max(S, [], 2);
[max_S, max_S_row_idx] = max(S_row_max);
A_col_max = max(A, [], 1);
[max_A, max_A_col_idx] = max(A_col_max);

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
    
    mean_val=mean(S,2);
    mean_mean_val=mean(mean_val);
    idx=find(mean_val<0.01*mean_mean_val);
    S(idx,:)=2*mean_mean_val*rand(length(idx),size(S,2));
    
    U = X*S';
    V = S*S';
    idx = randperm(r);
    for j_count = 1:r
        j = idx(j_count);
        W = A(:,j)+(U(:,j)-A*V(:,j))/V(j,j);
        W = max(W, eps_val*max_A);
        A(:,j) = W;
        
        A_col_max(j) = max(W);
        if A_col_max(j) >= max_A
            max_A = A_col_max(j);
            max_A_col_idx = j;
        elseif j == max_A_col_idx
            [max_A, max_A_col_idx] = max(A_col_max);
        end
    end
    
% %    temp = sqrt(sum(A.^2,1));
%     A = bsxfun(@rdivide, A, temp);
%     S = bsxfun(@times, S, temp');
    A(end,:)=delta*ones(1,size(A,2),class_type);
    
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
A(end,:)=[];

end

