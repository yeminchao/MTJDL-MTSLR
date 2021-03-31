function [model,classes]=mc_logistic_L12_train(X, Y, lambda, varargin)

classes=unique(Y{1});
K=length(classes);
T=length(Y);



if ~isempty(varargin)
    opts.init=1;
end
opts.maxIter=2000;
opts.tFlag=4;
opts.tol=1e-6;
opts.nFlag=0;
opts.rFlag=0;
opts.q=2;
opts.mFlag=0;
opts.lFlag=0;


model = repmat(struct, 1, K*(K-1)/2);

for i=1:K
    for j=1:i-1
        for t=1:T
            sample_idx=find(Y{t}==classes(i)|Y{t}==classes(j));
            Y_temp=Y{t}(sample_idx);
            pos_idx=Y_temp==classes(i);
            neg_idx=Y_temp==classes(j);
            Y_temp(pos_idx)=1;
            Y_temp(neg_idx)=-1;
            Y_train{t}=Y_temp;
            X_train{t}=X{t}(sample_idx,:);
        end
        idx=(i-2)*(i-1)/2+j;
        
        
        
        %                 [model(idx).W,model(idx).c]=Logistic_L21(X_train, Y_train, lambda, opts);
        opts.ind=0;
        for t=1:T
            opts.ind(t+1)=opts.ind(t)+size(X_train{t},1);
        end
        
        if ~isempty(varargin)
            model0=varargin{1};
            opts.x0=model0(idx).W;
            opts.c0=model0(idx).c;
        end
%         tic;
        [model(idx).W,model(idx).c, funVal, ValueL]=mtLogisticR(cat(1,X_train{:}), cat(1,Y_train{:}), lambda, opts);
%         toc;
        
    end
end

























