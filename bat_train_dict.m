function bat_train_dict

m=5:30;

parfor i=1:length(m)
%     for j=1:length(rho)
%         train_dict(m(i),rho(j));
%     end
  train_dict(m(i));
end



