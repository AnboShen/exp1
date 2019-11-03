data = csvread('diabetes.csv');
[ flist  W] = spfs_lar(data(:,1:8) , data(:,9) , 2);
selected_data = data(:,flist);
train_lable = data(1:700,9);
predict_lable=data(701:768,9);
train_data = selected_data(1:700,:);
predict_data = selected_data(701:768,:);

[bestacc,bestc,bestg] = SVMcg(train_lable,train_data,-8,8,-12,12,3,1,1,1.5);
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(train_lable,train_data,cmd);
[predict_label, accuracy, dec_values] = svmpredict(predict_lable, predict_data, model);