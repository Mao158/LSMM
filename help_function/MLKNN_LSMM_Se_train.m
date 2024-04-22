function [Prior, PriorN, Cond, CondN] = MLKNN_LSMM_Se_train(train_data, train_target, para, L)

[num_label, num_train] = size(train_target);
dim_data = size(train_data, 2);
dim_metric = size(L,2) / dim_data;
num_neighbour = para.num_MLKNN_neighbour;
smooth = para.smooth;

% Computing the prior probability
Prior = (sum(train_target, 2) + smooth) ./ (2 * smooth + num_train);
PriorN = 1 - Prior;

% Identifying k-nearest neighbors under different metrics
% computing distance between instances
neighbours = cell(num_label, 1);
for i = 1 : num_label
    ins_pos = train_target(i,:) == 1;
    ins_neg = train_target(i,:) == 0;

    curr_pos_L = mat((L(2*i-1,:)' + L(end,:)'), size(train_data, 2));
    curr_neg_L = mat((L(2*i,:)' + L(end,:)'), size(train_data, 2));

    % ascertain K-neighbours of each train data
    K_neighbour_index = zeros(num_train, num_neighbour);
    for j = 1:num_train
        dis_temp = train_data - repmat(train_data(j,:), num_train, 1);
        dis_L = zeros(num_train, dim_metric);
        dis_L(ins_pos,:) = dis_temp(ins_pos,:) * curr_pos_L;
        dis_L(ins_neg,:) = dis_temp(ins_neg,:) * curr_neg_L;
        dis = sum(dis_L.^2, 2);
        clear dis_temp dis_L
        [~, sorted_indices] = sort(dis, 'ascend');
        K_neighbour_index(j,:) = sorted_indices(2:num_neighbour + 1)';% delete itself which is located in the first column of matrix 'K_neighbour_index'
        clear dis sorted_indices
    end
    neighbours{i} = K_neighbour_index;
end


% Computing the likelihood
Cond = zeros(num_label, num_neighbour + 1);
CondN = zeros(num_label, num_neighbour + 1);
for j = 1 : num_label
    temp_Cj = zeros(num_neighbour + 1, 1); % The number of instances belong to the jth label which has k nearest neighbors belonging to the jth label is stored in temp_Cj(k+1)
    temp_NCj = zeros(num_neighbour + 1, 1); % The number of instances does not belong to the jth class which has k nearest neighbors belonging to the jth class is stored in temp_NCj(k+1)

    for i = 1 : num_train
        temp_k = sum(train_target(j, neighbours{j}(i, :))); % temp_k nearest neightbors of the ith instance belong to the jth class
        if (train_target(j, i) == 1)
            temp_Cj(temp_k + 1) = temp_Cj(temp_k + 1) + 1;
        else
            temp_NCj(temp_k + 1) = temp_NCj(temp_k + 1) + 1;
        end
    end

    sum_Cj = sum(temp_Cj);
    sum_NCj = sum(temp_NCj);
    for k = 1 : (num_neighbour + 1)
        Cond(j, k) = (smooth + temp_Cj(k)) / ((num_neighbour + 1) * smooth + sum_Cj);
        CondN(j, k) = (smooth + temp_NCj(k)) / ((num_neighbour + 1) * smooth + sum_NCj);
    end
end




