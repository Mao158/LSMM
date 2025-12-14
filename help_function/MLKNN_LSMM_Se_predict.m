% function [Outputs, Pre_Labels] = MLKNN_LSMM_Se_predict(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, L)
% 
% [num_test, dim_data] = size(test_data);
% num_train = size(train_data, 1);
% num_label = size(train_target, 1);
% dim_metric = size(L,2) / dim_data;
% 
% % Identifying k-nearest neighbors under different metrics
% % computing distance between testing instances and training instances
% neighbours = cell(num_label, 1);
% for i = 1 : num_label
%     ins_pos = train_target(i,:) == 1;
%     ins_neg = train_target(i,:) == 0;
% 
%     curr_pos_L = mat((L(2*i-1,:)' + L(end,:)'), size(train_data, 2));
%     curr_neg_L = mat((L(2*i,:)' + L(end,:)'), size(train_data, 2));
% 
%     % ascertain K-neighbours of each train data
%     K_neighbour_index = zeros(num_test, num_neighbour);
%     for j = 1:num_test
%         dis_temp = train_data - repmat(test_data(j,:), num_train, 1);
%         dis_L = zeros(num_train, dim_metric);
%         dis_L(ins_pos,:) = dis_temp(ins_pos,:) * curr_pos_L;
%         dis_L(ins_neg,:) = dis_temp(ins_neg,:) * curr_neg_L;
%         dis = sum(dis_L.^2, 2);
%         clear dis_temp dis_L
%         [~, sorted_indices] = sort(dis, 'ascend');
%         K_neighbour_index(j,:) = sorted_indices(1:num_neighbour)';
%         clear dis sorted_indices
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

function [Outputs, Pre_Labels] = MLKNN_LSMM_Se_predict(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, L)

%% ===== Basic sizes =====
[num_test, dim_data] = size(test_data);
num_train = size(train_data, 1);
num_label = size(train_target, 1);
dim_metric = size(L,2) / dim_data;

%% ===== Neighbor search (STRICTLY identical geometry) =====
neighbours = cell(num_label, 1);

for j = 1:num_label
    % positive / negative split (identical)
    ins_pos = (train_target(j,:) == 1);
    ins_neg = (train_target(j,:) == 0);

    % separable metrics (local + global, identical)
    curr_pos_L = mat((L(2*j-1,:)' + L(end,:)'), dim_data);
    curr_neg_L = mat((L(2*j,:)'   + L(end,:)'), dim_data);

    K_neighbour_index = zeros(num_test, num_neighbour);

    % ---- per-test-sample loop (KEEP) ----
    for i = 1:num_test
        % EXACT replacement of repmat
        diff = train_data - test_data(i,:);   % num_train x dim_data

        % EXACT separable metric application
        dis_L = zeros(num_train, dim_metric);
        dis_L(ins_pos,:) = diff(ins_pos,:) * curr_pos_L;
        dis_L(ins_neg,:) = diff(ins_neg,:) * curr_neg_L;

        % EXACT squared distance
        dis = sum(dis_L.^2, 2);

        % EXACT sorting (NO self-removal here)
        [~, idx] = sort(dis, 'ascend');
        K_neighbour_index(i,:) = idx(1:num_neighbour)';
    end

    neighbours{j} = K_neighbour_index;
end

%% ===== Probability inference (vectorized, STRICTLY equivalent) =====
Outputs = zeros(num_label, num_test);

for j = 1:num_label
    neigh_j = neighbours{j};        % num_test x K
    label_j = train_target(j,:);    % 1 x num_train

    % EXACT k-count definition
    k_count = sum(label_j(neigh_j), 2);   % num_test x 1

    % EXACT probability computation
    p_in  = Prior(j)  * Cond(j,  k_count + 1);
    p_out = PriorN(j) * CondN(j, k_count + 1);

    Outputs(j,:) = (p_in ./ (p_in + p_out))';
end

%% ===== Decision =====
Pre_Labels = ones(num_label, num_test);
Pre_Labels(Outputs <= 0.5) = -1;

end

