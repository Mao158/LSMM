function hammingloss = HammingLoss(Pre_Labels,test_target)
% Computing the hamming loss
% Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
% test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
%
% By: Jun-Xiang Mao
% Data: 2023.11.7

Pre_Labels(Pre_Labels <= 0) = -1;
test_target(test_target <= 0) = -1;

[num_label,num_test] = size(Pre_Labels);
miss_pairs = sum(sum(Pre_Labels ~= test_target));
hammingloss = miss_pairs / (num_label * num_test);
end