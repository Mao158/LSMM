function [macroAUC] = MacroAUC(Outputs,test_target)
% Computing the Macro AUC
% Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
% test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
%
% By: Jun-Xiang Mao
% Data: 2023.11.7

test_target(test_target<=0) = -1;

test_target(test_target<=0) = 0;
positive_labels = any(test_target, 2);
Outputs = Outputs(positive_labels, :);
test_target = test_target(positive_labels, :);

[num_labels, ~] = size(Outputs);

label_AUCs = zeros(1, num_labels);

% calculate AUC of each label
for label = 1:num_labels
    [~, ~, ~, AUC] = perfcurve(test_target(label, :), Outputs(label, :), 1);
    label_AUCs(label) = AUC;
end

% calculate Macro AUC
macroAUC = mean(label_AUCs);

% [num_label,num_data] = size(Outputs);
% macroAUC = 0;
% valid_label = num_label;
% 
% for i = 1:num_label
%     positive_idx = find(test_target(i,:) == 1);
%     negative_idx = setdiff([1:num_data],positive_idx);
%     num_positive = size(positive_idx,2);
%     num_negative = size(negative_idx,2);
% 
%     if num_positive == 0 || num_negative == 0
%         valid_label = valid_label - 1;
%         continue;
%     end
% 
%     AUC = 0;
% 
%     for pp = 1 : num_positive
%         for  nn = 1 : num_negative
%             if Outputs(i,positive_idx(1,pp)) > Outputs(i,negative_idx(1,nn))
%                 AUC = AUC + 1;
%             elseif Outputs(i,positive_idx(1,pp)) == Outputs(i,negative_idx(1,nn))
%                 AUC = AUC + 0.5;
%             end
%         end
%     end
% 
% AUC = AUC / (num_positive * num_negative);
% macroAUC = macroAUC + AUC;
% 
% end
% macroAUC = macroAUC / valid_label;
end

