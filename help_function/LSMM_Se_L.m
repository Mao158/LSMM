function [L, obj] = LSMM_Se_L(train_data, train_target, para, fold)
% Label-Specific Multi-semantics Metric Learning for Multi-Label Data 
% (transformation L)

num_label = size(train_target,1); % number of label
dim_data = size(train_data,2); % dimension of data
num_positive = para.num_positive;
num_negative = para.num_negative;
dim_reduce = para.dim_reduce;
max_iter = para.max_iter;

% Generate label-specific side information w.r.t each label
side_info = cell(num_label,1);
for i = 1:num_label
    Tri = generate_knntriplets(train_data, train_target(i,:)', num_positive, num_negative);
    must_link = unique(Tri(:,[1,2]), 'rows');% postive pairs
    cannot_link = unique(Tri(:,[1,3]), 'rows');% negative pairs
    clear Tri;
    side_info{i,1} = [[must_link, ones(size(must_link, 1), 1)]; [cannot_link, -ones(size(cannot_link, 1), 1)]];
    clear must_link cannot_link;

    % Generate positve/negative instance information
    num_info = size(side_info{i,1},1);
    ins_info = zeros(num_info, 1);
    ins_info(train_target(i,side_info{i,1}(:,1)) == 1) = 1;
    side_info{i,1} = [side_info{i,1}, ins_info];
    clear ins_info;
end

% initialize local and global metric
% each metric is stored in the manner of row vector
dim_metric = ceil(dim_data * (1 - dim_reduce)); % dimension of the learned metrics
num_metric = 2 * num_label + 1; % number of metric (2 local metrics per label + 1 global metric)
L = repmat(vec(zeros(dim_data, dim_metric))', num_metric - 1, 1);% initialize local metrics with zero matrix
L = [L; vec(rand(dim_data, dim_metric))'];% initialize the global metric

L = vec(L);

% L-BFGS optimization
optim = optimLineSearch();
options = struct('maxIter',max_iter,'Display','iter-detailed','TolX',1e-5,'TolFun',1e-5);
fun = @(L) compute_Se_grad_L(L, train_data, side_info, para);
[L, obj] = optim.runOptimLineSearch(fun, L, options, para, fold);

% Adaptive learning rate gradient descent optimization
% [L, obj] = Se_GD(L, train_data, side_info, para, fold);

L = mat(L, num_metric);



% % compute the average distance of all positive pairs and negative pairs with learned metric
% positive_ave = zeros(num_label,1);
% negative_ave = zeros(num_label,1);
% positive_pair_dist = cell(num_label,1);
% negative_pair_dist = cell(num_label,1);
% for kk = 1:num_label
%     if para.with_global == true
%         current_L = mat(L(kk,:)') + mat(L(end,:)');
%     else
%         current_L = mat(L(kk,:)');
%     end
%     current_M = current_L*current_L';
%     [positive_ave(kk,1),negative_ave(kk,1),positive_pair_dist{kk,1},negative_pair_dist{kk,1}] = ave(current_M,side_info{kk},train_data);
% end


