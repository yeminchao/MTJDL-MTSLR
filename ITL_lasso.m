function ITL_lasso(m)


if ~exist(fullfile(pwd,'data'),'dir')
    mkdir(fullfile(pwd,'data'));
end

trainData1=[];
trainLabel1=[];
trainData2=[];
trainLabel2=[];
testData2=[];

load('data/sample.mat','trainData1','trainLabel1','trainData2','trainLabel2',...
    'testData2','testLabel2','mean_val1','mean_val2','sigma_val1','sigma_val2');

trainData1=bsxfun(@plus,bsxfun(@times,trainData1,sigma_val1),mean_val1);
trainData2=bsxfun(@plus,bsxfun(@times,trainData2,sigma_val2),mean_val2);
testData2=bsxfun(@plus,bsxfun(@times,testData2,sigma_val2),mean_val2);

load(fullfile('data',sprintf('dictionary_m=%d.mat',m)),'D');

trainData1=NNLS(D, trainData1')';
trainData2=NNLS(D, trainData2')';
testData2=NNLS(D, testData2')';

lambda=10.^(-1:-1:-7);

for lambda_count=1:length(lambda)
    lambda_value=lambda(lambda_count);
    if lambda_count==1
        [model_mt,classes_mt]=mc_logistic_L12_train({trainData1,trainData2}, {trainLabel1,trainLabel2}, lambda_value);
    else
        [model_mt,classes_mt]=mc_logistic_L12_train({trainData1,trainData2}, {trainLabel1,trainLabel2}, lambda_value, model_mt);
    end
    predictLabel_mt=mc_logistic_predict(model_mt,classes_mt,2,testData2);
    oa_mt=overall_accuracy(testLabel2,predictLabel_mt);
    aa_mt=average_accuracy(testLabel2,predictLabel_mt);
    kappa_mt=compute_kappa(testLabel2,predictLabel_mt);
    
    parsave(sprintf('data/result-ITL_m=%d_logistic_lambda=%.8g.mat',m,lambda_value),...
        lambda_value,predictLabel_mt,oa_mt,aa_mt,kappa_mt,model_mt,classes_mt);
end


