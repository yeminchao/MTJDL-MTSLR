function train_dict(m)

load('DataCube.mat','DataCube1','DataCube2');


max_sample_dict=100000;

X=[reshape(DataCube1,size(DataCube1,1)*size(DataCube1,2),size(DataCube1,3))',reshape(DataCube2,size(DataCube2,1)*size(DataCube2,2),size(DataCube2,3))'];

idx=randperm(size(X,2));
idx=idx(1:min(max_sample_dict,size(X,2)));
X=X(:,idx);

opts.maxit=500;

[D,~]=NMF(X, m, opts);



save(fullfile('data',sprintf('dictionary_m=%d.mat',m)),'D','m');




































