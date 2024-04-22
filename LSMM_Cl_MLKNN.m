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
% Parameters of K-means
para.num_cluster = 3;
% Parameters of MLKNN
para.num_MLKNN_neighbour = 10;
para.smooth = 1.0;

num_fold = para.num_fold;

Result_MLKNN = zeros(num_fold, 6);
Result_LSMMCL = zeros(num_fold, 6);

% Set a random seed to make the experiment reproducible
seed = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(seed);
indices = crossvalind('Kfold', num_data, 10);

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

    % K-means cluster
    cluster_idx = kmeans(train_data, para.num_cluster);

    % Compute label-specific multiple metrics for multi-label data
    [L, obj] = LSMM_Cl_L(train_data, train_target, para, fold, cluster_idx);

    % MLKNN
    [Prior_MLKNN, PriorN_MLKNN, Cond_MLKNN, CondN_MLKNN] = MLKNN_train_M(train_data, train_target, para, eye(num_dim));
    [Outputs_MLKNN, Pre_Labels_MLKNN] = MLKNN_predict_M(test_data, train_data, train_target, para.num_MLKNN_neighbour, Prior_MLKNN, PriorN_MLKNN, Cond_MLKNN, CondN_MLKNN, eye(num_dim));
    [HammingLoss_MLKNN, RankingLoss_MLKNN, Coverage_MLKNN, Average_Precision_MLKNN, MacroF1_MLKNN, MacroAUC_MLKNN] = MLEvaluate(Outputs_MLKNN, Pre_Labels_MLKNN, test_target);
    Result_MLKNN(fold,:) = [HammingLoss_MLKNN, RankingLoss_MLKNN, Coverage_MLKNN, Average_Precision_MLKNN, MacroF1_MLKNN, MacroAUC_MLKNN];

    % MLKNN is coupled with LSMMCL: MLKNN-LSMMCL
    [Prior_LSMMCL, PriorN_LSMMCL, Cond_LSMMCL, CondN_LSMMCL] = MLKNN_LSMM_Cl_train(train_data, train_target, para, cluster_idx, L);
    [Outputs_LSMMCL, Pre_Labels_LSMMCL] = MLKNN_LSMM_Cl_predict(test_data, train_data, train_target, Prior_LSMMCL, PriorN_LSMMCL, Cond_LSMMCL, CondN_LSMMCL, para, cluster_idx, L);
    [HammingLoss_LSMMCL, RankingLoss_LSMMCL, Coverage_LSMMCL, Average_Precision_LSMMCL, MicroF1_LSMMCL, MacroAUC_LSMMCL] = MLEvaluate(Outputs_LSMMCL, Pre_Labels_LSMMCL, test_target);
    Result_LSMMCL(fold,:) = [HammingLoss_LSMMCL, RankingLoss_LSMMCL, Coverage_LSMMCL, Average_Precision_LSMMCL, MicroF1_LSMMCL, MacroAUC_LSMMCL];
     
end
Result_MLKNN_mean = round(mean(Result_MLKNN,1),3);
Result_MLKNN_std = round(std(Result_MLKNN,0,1),3);
Result_LSMMCL_mean = round(mean(Result_LSMMCL,1),3);
Result_LSMMCL_std = round(std(Result_LSMMCL,0,1),3);

% Print results of MLKNN
fprintf('MLKNN results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_MLKNN_mean(1), Result_MLKNN_std(1), Result_MLKNN_mean(2), Result_MLKNN_std(2), ...
    Result_MLKNN_mean(3), Result_MLKNN_std(3), Result_MLKNN_mean(4), Result_MLKNN_std(4), Result_MLKNN_mean(5), Result_MLKNN_std(5), Result_MLKNN_mean(6), Result_MLKNN_std(6));

% Print results of MLKNN-LSMMCL
fprintf('MLKNN-LSMMCL results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_LSMMCL_mean(1), Result_LSMMCL_std(1), Result_LSMMCL_mean(2), Result_LSMMCL_std(2), ...
    Result_LSMMCL_mean(3), Result_LSMMCL_std(3), Result_LSMMCL_mean(4), Result_LSMMCL_std(4), Result_LSMMCL_mean(5), Result_LSMMCL_std(5), Result_LSMMCL_mean(6), Result_LSMMCL_std(6));


