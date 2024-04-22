function [Outputs, Pre_Labels] = MLKNN_LSMM_Cl_predict(test_data, train_data, train_target, Prior, PriorN, Cond, CondN, para, cluster_idx, L)

[num_test, dim_data] = size(test_data);
num_train = size(train_data, 1);
num_label = size(train_target, 1);
dim_metric = size(L,2) / dim_data;
K = para.num_MLKNN_neighbour; % parameter of MLKNN
num_cluster = para.num_cluster;

% Identifying k-nearest neighbors under different metrics
% computing distance between testing instances and training instances
neighbours = cell(num_label, 1);
for i = 1 : num_label
    indices_cluster_idx = (num_cluster * (i - 1) + 1) : (num_cluster *i);
    cluster_L = arrayfun(@(j) mat((L(indices_cluster_idx(j), :)' + L(end, :)'), size(train_data, 2)), 1:num_cluster, 'UniformOutput', false);
    
    % ascertain K-neighbours of each train data
    K_neighbour_index = zeros(num_test, K);
    for j = 1:num_test
        dis_temp = train_data - repmat(test_data(j,:), num_train, 1);
        dis_L = zeros(num_train, dim_metric);
        for k = 1:num_cluster
            cluster_ins_indices = find(cluster_idx == k);
            dis_L(cluster_ins_indices,:) = dis_temp(cluster_ins_indices,:) *cluster_L{k};
        end
        dis = sum(dis_L.^2, 2);
        clear dis_temp dis_L
        [~, sorted_indices] = sort(dis, 'ascend');
        K_neighbour_index(j,:) = sorted_indices(1:K)';
    end
    neighbours{i} = K_neighbour_index;
end

% Computing probability
Outputs = zeros(num_label, num_test);
prob_in = zeros(num_label, 1); % The probability P[Hj]*P[k|Hj] is stored in prob_in(j)
prob_out = zeros(num_label, 1); % The probability P[~Hj]*P[k|~Hj] is stored in prob_out(j)
for i = 1:num_test
    temp_C = zeros(num_label, 1);
    for j = 1:num_label
        temp_C(j) = sum((train_target(j, neighbours{j}(i, :))), 2); % The number of nearest neighbors belonging to the jth class is stored in temp_C(j, 1)
        prob_in(j) = Prior(j) * Cond(j, temp_C(j) + 1);
        prob_out(j) = PriorN(j) * CondN(j, temp_C(j) + 1);
    end
    Outputs(:, i) = prob_in ./ (prob_in + prob_out);
end

% Assigning labels for testing instances
Pre_Labels = ones(num_label, num_test);
Pre_Labels(Outputs <= 0.5) = -1;
