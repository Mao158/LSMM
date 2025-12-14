% function [Prior, PriorN, Cond, CondN] = MLKNN_LSMM_Cl_train(train_data, train_target, para, cluster_idx, L)
% 
% [num_label, num_train] = size(train_target);
% dim_data = size(train_data, 2);
% dim_metric = size(L,2) / dim_data;
% num_neighbour = para.num_MLKNN_neighbour;
% smooth = para.smooth;
% num_cluster = para.num_cluster;
% 
% % Computing the prior probability
% Prior = (sum(train_target, 2) + smooth) ./ (2 * smooth + num_train);
% PriorN = 1 - Prior;
% 
% % Identifying k-nearest neighbors under different metrics
% % computing distance between instances
% neighbours = cell(num_label, 1);
% for i = 1 : num_label
%     indices_cluster_idx = (num_cluster * (i - 1) + 1) : (num_cluster *i);
%     cluster_L = arrayfun(@(j) mat((L(indices_cluster_idx(j), :)' + L(end, :)'), size(train_data, 2)), 1:num_cluster, 'UniformOutput', false);
% 
%     % ascertain K-neighbours of each train data
%     K_neighbour_index = zeros(num_train, num_neighbour);
%     for j = 1:num_train
%         dis_temp = train_data - repmat(train_data(j,:), num_train, 1);
%         dis_L = zeros(num_train, dim_metric);
%         for k = 1:num_cluster
%             cluster_ins_indices = find(cluster_idx == k);
%             dis_L(cluster_ins_indices,:) = dis_temp(cluster_ins_indices,:) *cluster_L{k};
%         end
%         dis = sum(dis_L.^2, 2);
%         clear dis_temp dis_L
%         [~, sorted_indices] = sort(dis, 'ascend');
%         K_neighbour_index(j,:) = sorted_indices(2:num_neighbour + 1)';% delete itself which is located in the first column of matrix 'K_neighbour_index'
%         clear dis sorted_indices
%     end
%     neighbours{i} = K_neighbour_index;
% end
% 
% 
% % Computing the likelihood
% Cond = zeros(num_label, num_neighbour + 1);
% CondN = zeros(num_label, num_neighbour + 1);
% for j = 1 : num_label
%     temp_Cj = zeros(num_neighbour + 1, 1); % The number of instances belong to the jth label which has k nearest neighbors belonging to the jth label is stored in temp_Cj(k+1)
%     temp_NCj = zeros(num_neighbour + 1, 1); % The number of instances does not belong to the jth class which has k nearest neighbors belonging to the jth class is stored in temp_NCj(k+1)
% 
%     for i = 1 : num_train
%         temp_k = sum(train_target(j, neighbours{j}(i, :))); % temp_k nearest neightbors of the ith instance belong to the jth class
%         if (train_target(j, i) == 1)
%             temp_Cj(temp_k + 1) = temp_Cj(temp_k + 1) + 1;
%         else
%             temp_NCj(temp_k + 1) = temp_NCj(temp_k + 1) + 1;
%         end
%     end
% 
%     sum_Cj = sum(temp_Cj);
%     sum_NCj = sum(temp_NCj);
%     for k = 1 : (num_neighbour + 1)
%         Cond(j, k) = (smooth + temp_Cj(k)) / ((num_neighbour + 1) * smooth + sum_Cj);
%         CondN(j, k) = (smooth + temp_NCj(k)) / ((num_neighbour + 1) * smooth + sum_NCj);
%     end
% end

function [Prior, PriorN, Cond, CondN] = MLKNN_LSMM_Cl_train(train_data, train_target, para, cluster_idx, L)

%% ===== Basic sizes =====
[num_label, num_train] = size(train_target);
dim_data   = size(train_data, 2);
dim_metric = size(L,2) / dim_data;

num_neighbour = para.num_MLKNN_neighbour;
smooth        = para.smooth;
num_cluster   = para.num_cluster;

%% ===== Prior (IDENTICAL) =====
Prior  = (sum(train_target, 2) + smooth) ./ (2*smooth + num_train);
PriorN = 1 - Prior;

Cond  = zeros(num_label, num_neighbour + 1);
CondN = zeros(num_label, num_neighbour + 1);

%% ===== Precompute cluster indices (SAFE optimization) =====
cluster_indices = cell(num_cluster,1);
for k = 1:num_cluster
    cluster_indices{k} = find(cluster_idx == k);
end

%% ===== Neighbor search (STRICTLY identical geometry) =====
neighbours = cell(num_label, 1);

for i = 1:num_label
    % build cluster-specific metrics (IDENTICAL)
    indices_cluster_idx = (num_cluster*(i-1)+1):(num_cluster*i);
    cluster_L = cell(num_cluster,1);
    for k = 1:num_cluster
        cluster_L{k} = mat((L(indices_cluster_idx(k), :)' + L(end,:)'), dim_data);
    end

    K_neighbour_index = zeros(num_train, num_neighbour);

    % ---- per-train-sample loop (KEEP) ----
    for j = 1:num_train
        % EXACT replacement of repmat
        diff = train_data - train_data(j,:);   % num_train x dim_data

        % EXACT cluster-wise metric application
        dis_L = zeros(num_train, dim_metric);
        for k = 1:num_cluster
            idx_k = cluster_indices{k};
            dis_L(idx_k,:) = diff(idx_k,:) * cluster_L{k};
        end

        % EXACT squared distance
        dis = sum(dis_L.^2, 2);

        % EXACT sorting & self-removal
        [~, idx] = sort(dis, 'ascend');
        K_neighbour_index(j,:) = idx(2:num_neighbour+1)';
    end

    neighbours{i} = K_neighbour_index;
end

%% ===== Likelihood (STRICTLY equivalent, vectorized) =====
for j = 1:num_label
    neigh_j = neighbours{j};        % num_train x K
    label_j = train_target(j,:);    % 1 x num_train

    % EXACT k-count definition
    k_count = sum(label_j(neigh_j), 2);

    pos = (label_j' == 1);
    neg = ~pos;

    temp_Cj  = accumarray(k_count(pos)  + 1, 1, ...
                          [num_neighbour+1,1], @sum, 0);
    temp_NCj = accumarray(k_count(neg) + 1, 1, ...
                          [num_neighbour+1,1], @sum, 0);

    Cond(j,:)  = (temp_Cj'  + smooth) ./ ...
                 (sum(temp_Cj)  + (num_neighbour+1)*smooth);
    CondN(j,:) = (temp_NCj' + smooth) ./ ...
                 (sum(temp_NCj) + (num_neighbour+1)*smooth);
end

end




