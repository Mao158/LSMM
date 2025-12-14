% function [Outputs, Pre_Labels] = MLKNN_LSMM_Cl_predict(test_data, train_data, train_target, Prior, PriorN, Cond, CondN, para, cluster_idx, L)
% 
% [num_test, dim_data] = size(test_data);
% num_train = size(train_data, 1);
% num_label = size(train_target, 1);
% dim_metric = size(L,2) / dim_data;
% K = para.num_MLKNN_neighbour; % parameter of MLKNN
% num_cluster = para.num_cluster;
% 
% % Identifying k-nearest neighbors under different metrics
% % computing distance between testing instances and training instances
% neighbours = cell(num_label, 1);
% for i = 1 : num_label
%     indices_cluster_idx = (num_cluster * (i - 1) + 1) : (num_cluster *i);
%     cluster_L = arrayfun(@(j) mat((L(indices_cluster_idx(j), :)' + L(end, :)'), size(train_data, 2)), 1:num_cluster, 'UniformOutput', false);
% 
%     % ascertain K-neighbours of each train data
%     K_neighbour_index = zeros(num_test, K);
%     for j = 1:num_test
%         dis_temp = train_data - repmat(test_data(j,:), num_train, 1);
%         dis_L = zeros(num_train, dim_metric);
%         for k = 1:num_cluster
%             cluster_ins_indices = find(cluster_idx == k);
%             dis_L(cluster_ins_indices,:) = dis_temp(cluster_ins_indices,:) *cluster_L{k};
%         end
%         dis = sum(dis_L.^2, 2);
%         clear dis_temp dis_L
%         [~, sorted_indices] = sort(dis, 'ascend');
%         K_neighbour_index(j,:) = sorted_indices(1:K)';
%     end
%     neighbours{i} = K_neighbour_index;
% end
% 
% % Computing probability
% Outputs = zeros(num_label, num_test);
% prob_in = zeros(num_label, 1); % The probability P[Hj]*P[k|Hj] is stored in prob_in(j)
% prob_out = zeros(num_label, 1); % The probability P[~Hj]*P[k|~Hj] is stored in prob_out(j)
% for i = 1:num_test
%     temp_C = zeros(num_label, 1);
%     for j = 1:num_label
%         temp_C(j) = sum((train_target(j, neighbours{j}(i, :))), 2); % The number of nearest neighbors belonging to the jth class is stored in temp_C(j, 1)
%         prob_in(j) = Prior(j) * Cond(j, temp_C(j) + 1);
%         prob_out(j) = PriorN(j) * CondN(j, temp_C(j) + 1);
%     end
%     Outputs(:, i) = prob_in ./ (prob_in + prob_out);
% end
% 
% % Assigning labels for testing instances
% Pre_Labels = ones(num_label, num_test);
% Pre_Labels(Outputs <= 0.5) = -1;

function [Outputs, Pre_Labels] = MLKNN_LSMM_Cl_predict(test_data, train_data, train_target, Prior, PriorN, Cond, CondN, para, cluster_idx, L)

%% ===== Basic sizes =====
[num_test, dim_data] = size(test_data);
num_train = size(train_data, 1);
num_label = size(train_target, 1);
dim_metric = size(L,2) / dim_data;

K = para.num_MLKNN_neighbour;
num_cluster = para.num_cluster;

%% ===== Precompute cluster indices (SAFE) =====
cluster_indices = cell(num_cluster,1);
for k = 1:num_cluster
    cluster_indices{k} = find(cluster_idx == k);
end

%% ===== Neighbor search (STRICTLY identical geometry) =====
neighbours = cell(num_label, 1);

for i = 1:num_label
    % cluster-specific metrics (IDENTICAL)
    indices_cluster_idx = (num_cluster*(i-1)+1):(num_cluster*i);
    cluster_L = cell(num_cluster,1);
    for k = 1:num_cluster
        cluster_L{k} = mat((L(indices_cluster_idx(k), :)' + L(end,:)'), dim_data);
    end

    K_neighbour_index = zeros(num_test, K);

    % ---- per-test loop (KEEP) ----
    for j = 1:num_test
        % EXACT replacement of repmat
        diff = train_data - test_data(j,:);   % num_train x dim_data

        % EXACT cluster-wise metric application
        dis_L = zeros(num_train, dim_metric);
        for k = 1:num_cluster
            idx_k = cluster_indices{k};
            dis_L(idx_k,:) = diff(idx_k,:) * cluster_L{k};
        end

        % EXACT squared distance
        dis = sum(dis_L.^2, 2);

        % EXACT sorting (NO self-removal)
        [~, idx] = sort(dis, 'ascend');
        K_neighbour_index(j,:) = idx(1:K)';
    end

    neighbours{i} = K_neighbour_index;
end

%% ===== Probability inference (vectorized, STRICTLY equivalent) =====
Outputs = zeros(num_label, num_test);

for j = 1:num_label
    neigh_j = neighbours{j};        % num_test x K
    label_j = train_target(j,:);    % 1 x num_train

    % EXACT k-count definition
    k_count = sum(label_j(neigh_j), 2);   % num_test x 1

    % EXACT posterior
    p_in  = Prior(j)  * Cond(j,  k_count + 1);
    p_out = PriorN(j) * CondN(j, k_count + 1);

    Outputs(j,:) = (p_in ./ (p_in + p_out))';
end

%% ===== Decision =====
Pre_Labels = ones(num_label, num_test);
Pre_Labels(Outputs <= 0.5) = -1;

end
