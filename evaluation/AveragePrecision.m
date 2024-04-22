function AP = AveragePrecision(Outputs,test_target)
% Computing the average precision
% Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
% test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
%
% By: Jun-Xiang Mao
% Data: 2023.11.7

test_target(test_target <= 0) = -1;

num_label = size(Outputs,1);
index = (sum(test_target) ~= num_label) & (sum(test_target) ~= -num_label);
temp_Outputs = Outputs(:,index);
temp_test_target = test_target(:,index);
[~,num_test] = size(temp_Outputs);

AP = 0;
for i = 1:num_test
    temp_avg = 0;
    rel_l = find(temp_test_target(:,i) > 0);
    for j = rel_l'
        temp_avg = temp_avg + mean(temp_test_target(temp_Outputs(:,i) >= temp_Outputs(j,i),i) == 1);
    end
    AP = AP + temp_avg / length(rel_l);
end
AP = AP / num_test;
end