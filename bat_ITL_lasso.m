function bat_ITL_lasso

files=dir(fullfile('data','dictionary_m=*.mat'));

parfor i=1:length(files)
    func(fullfile('data',files(i).name));    
end

function func(file_name)
load(file_name,'m');
ITL_lasso(m);















