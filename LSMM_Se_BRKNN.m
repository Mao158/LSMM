clc;
clear;

addpath(genpath('dataset'));
addpath(genpath('evaluation'));
addpath(genpath('help_function'));
addpath(genpath('matlab-lbfgs'));

load('emotions.mat'); data_str = 'emotions';%593,72,6
% load('birds.mat'); data_str = 'birds';%645,258,19
% load('medical.mat'); data_str = 'medical';%978,1449,45
% load('enron.mat'); data_str = 'enron';%1702,1001,53
% load('image.mat'); data_str = 'image';%2000,294,5
% load('scene.mat'); data_str = 'scene';%2407,294,6
% load('slashdot.mat'); data_str = 'slashdot';%3781,1079,22
% load('arts.mat'); data_str = 'arts';%5000,462,26
% load('education.mat'); data_str = 'education';%5000,550,33

if ~strcmp(data_str, 'emotions') && ~strcmp(data_str, 'birds')
    data = PCA(data);
end

[num_data, num_dim] = size(data);
num_label = size(target,1);
para.num_fold = 10; % number of fold
para.data_str = data_str;
para.num_positive = 20;% number of positives
para.num_negative = 20;% number of negatives
para.max_iter = 1000;
para.dim_reduce = 0; % The reduction ratio of the dimension of the learned metrics, range from [0,1)
para.gamma = 2;
para.alphfa = 0.4;
% Here, lambda_1 and lambda_2 should be tuned by model selection stratgies, such as 5-fold cross validation
para.lambda_1 = 100;
para.lambda_2 = 0.001;
% parameter of BRKNN
para.num_BRKNN_neighbour = 10; 

num_fold = para.num_fold;

Result_BRKNN = zeros(num_fold, 6);
Result_LSMMSE = zeros(num_fold, 6);

% Set a random seed to make the experiment reproducible
seed = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(seed);
indices = crossvalind('Kfold',num_data,10);

parfor fold = 1 : num_fold
    seed2 = RandStream('mt19937ar','Seed',1);
    RandStream.setGlobalStream(seed2);

    test_logical = (indices == fold);
    train_logical = ~ test_logical;
    train_data = data(train_logical,:);
    test_data = data(test_logical,:);
    train_target = target(:,train_logical);
    test_target = target(:,test_logical);

    num_train = size(train_data,1);
    sum_class = sum(train_target,2); % Determine how many positive instances in each label
    condition = (sum_class >= 2) & (sum_class <= num_train - 2);  % when encountering severe class-imbalance problem, we ignore the corresponding label.
    train_target = train_target(condition,:);
    test_target = test_target(condition,:);

    % Compute label-specific multiple metrics for multi-label data
    [L, obj] = LSMM_Se_L(train_data, train_target, para, fold);

    % BRKNN
    [Outputs_BRKNN, Pre_Labels_BRKNN] = BRKNN(train_data, train_target, test_data, para);
    [HammingLoss_BRKNN,RankingLoss_BRKNN,Coverage_BRKNN,Average_Precision_BRKNN,MacroF1_BRKNN,MacroAUC_BRKNN] = MLEvaluate(Outputs_BRKNN,Pre_Labels_BRKNN,test_target);
    Result_BRKNN(fold,:) = [HammingLoss_BRKNN,RankingLoss_BRKNN,Coverage_BRKNN,Average_Precision_BRKNN,MacroF1_BRKNN,MacroAUC_BRKNN];

    % BRKNN is coupled with LSMMSE: BRKNN-LSMMSE
    [Outputs_LSMMSE, Pre_Labels_LSMMSE] = BRKNN_LSMM_Se_predict(train_data, train_target, test_data, para, L);
    [HammingLoss_LSMMSE,RankingLoss_LSMMSE,Coverage_LSMMSE,Average_Precision_LSMMSE,MacroF1_LSMMSE,MacroAUC_LSMMSE] = MLEvaluate(Outputs_LSMMSE,Pre_Labels_LSMMSE,test_target);
    Result_LSMMSE(fold,:) = [HammingLoss_LSMMSE,RankingLoss_LSMMSE,Coverage_LSMMSE,Average_Precision_LSMMSE,MacroF1_LSMMSE,MacroAUC_LSMMSE];
end
Result_BRKNN_mean = round(mean(Result_BRKNN,1),3);
Result_BRKNN_std = round(std(Result_BRKNN,0,1),3);
Result_LSMMSE_mean = round(mean(Result_LSMMSE,1),3);
Result_LSMMSE_std = round(std(Result_LSMMSE,0,1),3);

% Print results of BRKNN
fprintf('BRKNN results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_BRKNN_mean(1), Result_BRKNN_std(1), Result_BRKNN_mean(2), Result_BRKNN_std(2), ...
    Result_BRKNN_mean(3), Result_BRKNN_std(3), Result_BRKNN_mean(4), Result_BRKNN_std(4), Result_BRKNN_mean(5), Result_BRKNN_std(5), Result_BRKNN_mean(6), Result_BRKNN_std(6));

% Print results of BRKNN-LSMMSE
fprintf('BRKNN-LSMMSE results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_LSMMSE_mean(1), Result_LSMMSE_std(1), Result_LSMMSE_mean(2), Result_LSMMSE_std(2), ...
    Result_LSMMSE_mean(3), Result_LSMMSE_std(3), Result_LSMMSE_mean(4), Result_LSMMSE_std(4), Result_LSMMSE_mean(5), Result_LSMMSE_std(5), Result_LSMMSE_mean(6), Result_LSMMSE_std(6));





