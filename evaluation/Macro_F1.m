function MacroF1 = Macro_F1(Pre_Labels,test_target)
% Computing the Macro F1
% Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
% test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
%
% By: Jun-Xiang Mao
% Data: 2023.11.7

Pre_Labels(Pre_Labels<=0) = 0;
test_target(test_target<=0) = 0;

Pre_Labels = Pre_Labels > 0;
test_target = test_target > 0;

TP = Pre_Labels .* test_target;
U = Pre_Labels + test_target;

f = 2 * sum(TP,2) ./ sum(U,2);
f(isnan(f)) = [];
MacroF1 = mean(f);

% TP_temp = Pre_Labels&test_target; % true positive
% FP_temp = Pre_Labels&(~test_target); % false positive
% FN_temp = (~Pre_Labels)&test_target; % false negative
% 
% TP_matrix = sum(TP_temp,2);
% FP_matrix = sum(FP_temp,2);
% FN_matrix = sum(FN_temp,2);
% 
% [num_class,~]=size(Pre_Labels);
% 
% MacroF1=0;
% 
% for j=1:num_class
%     TP = TP_matrix(j);
%     FP = FP_matrix(j);
%     FN = FN_matrix(j);
%     if 2*TP+FN+FP == 0
%         F1 = 0;
%     else
%         F1 = 2*TP/(2*TP+FN+FP);
%     end
%     MacroF1 = MacroF1+F1;
% end
% MacroF1 = MacroF1/num_class;
end

