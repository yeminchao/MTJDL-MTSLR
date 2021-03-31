function predict_label=mc_logistic_predict(model,classes,t,X)
K=length(classes);
vote_arr=zeros(size(X,1),K);
for i=1:K
    for j=1:i-1
        idx=(i-2)*(i-1)/2+j;
        y = X*(model(idx).W(:, t)) + model(idx).c(t);
        p_pos=1./(1+exp(-y));
        vote_arr(:,i)=vote_arr(:,i)+p_pos;
        vote_arr(:,j)=vote_arr(:,j)+1-p_pos;
    end
end

[~,predict_label_idx]=max(vote_arr,[],2);
predict_label=classes(predict_label_idx);












