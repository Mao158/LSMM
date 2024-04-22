function [Outputs, Pre_Labels] = BRKNN_LSMM_Se_predict(train_data, train_target, test_data, para, L)

[num_test, dim_data] = size(test_data);
[num_label, num_train] = size(train_target);
dim_metric = size(L,2) / dim_data;
K = para.num_BRKNN_neighbour;% parameter of BRKNN
Outputs = zeros(num_label, num_test); % numerical-results([0,1]) of labels for each test instance
Pre_Labels = zeros(num_label, num_test); % logical-results({-1,1}) of labels for each test instance


for i = 1:num_label
    ins_pos = train_target(i,:) == 1;
    ins_neg = train_target(i,:) == 0;

    curr_pos_L = mat((L(2*i-1,:)' + L(end,:)'), size(train_data, 2));
    curr_neg_L = mat((L(2*i,:)' + L(end,:)'), size(train_data, 2));

    % ascertain K-neighbours of each test data
    K_neighbour_dist = zeros(num_test, K);
    K_neighbour_index = zeros(num_test, K);
    for j = 1:num_test
        dis_temp = train_data - repmat(test_data(j,:), num_train, 1);
        dis_L = zeros(num_train, dim_metric);
        dis_L(ins_pos,:) = dis_temp(ins_pos,:) * curr_pos_L;
        dis_L(ins_neg,:) = dis_temp(ins_neg,:) * curr_neg_L;
        dis = sum(dis_L.^2, 2);
        clear dis_temp dis_L
        [sorted_values, sorted_indices] = sort(dis, 'ascend');
        K_neighbour_dist(j,:) = sorted_values(1:K)';
        K_neighbour_index(j,:) = sorted_indices(1:K)';
    end

    K_neighbour_target_temp = train_target(i,K_neighbour_index);
    K_neighbour_target = reshape(K_neighbour_target_temp,[],K);
    % compute weight based on K-neighbour-distance
    % in the following way, the situation of K_neighbour_dist == 0 can be handled carefully 
    K_neighbour_dist_row = sum(K_neighbour_dist,2);
    Similarity_temp = bsxfun(@rdivide,K_neighbour_dist,K_neighbour_dist_row);
    Similarity_temp(isnan(Similarity_temp)) = 0.5;
    Similarity = ones(num_test,K) - Similarity_temp;
    sum_Similarity = sum(Similarity,2);
    Weight = bsxfun(@rdivide,Similarity,sum_Similarity);
    Outputs(i,:) = sum(Weight .* K_neighbour_target,2)';

    Pre_Labels(i,:) = Outputs(i,:);
    Pre_Labels(i,(Pre_Labels(i,:) >= 0.5)) = 1;
    Pre_Labels(i,(Pre_Labels(i,:) < 0.5)) = -1;
end