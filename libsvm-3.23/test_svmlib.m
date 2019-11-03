load('heart_scale.mat')
[bestacc,bestc,bestg] = SVMcg(heart_scale_label,heart_scale_inst,-5,5,-5,5,2,1,1,1.5);
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(heart_scale_label, heart_scale_inst,cmd);
[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label(136:270,:), heart_scale_inst(136:270,:), model);
