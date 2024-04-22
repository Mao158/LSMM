function [Outputs,Pre_Labels] = BRKNN(train_data, train_target, test_data, para)

num_BRKNN_neighbour = para.num_BRKNN_neighbour;
num_label = size(train_target,1);
num_test = size(test_data,1);

Outputs = zeros(num_label, num_test); % numerical-results([0,1]) of labels for each test instance
Pre_Labels = zeros(num_label, num_test); % logical-results({-1,1}) of labels for each test instance

for i = 1:num_label
    % ascertain K-neighbours of each test data

    [K_neighbour_index,K_neighbour_dist] = knnsearch(train_data,test_data,'K',num_BRKNN_neighbour);
    K_neighbour_dist = K_neighbour_dist .* K_neighbour_dist;

    K_neighbour_target_temp = train_target(i,K_neighbour_index);
    K_neighbour_target = reshape(K_neighbour_target_temp,[],num_BRKNN_neighbour);


    % compute weight based on K-neighbour-distance
    % in the following way, the situation of K_neighbour_dist == 0 can be handled carefully
    K_neighbour_dist_row = sum(K_neighbour_dist,2);
    Similarity_temp = bsxfun(@rdivide,K_neighbour_dist,K_neighbour_dist_row);
    Similarity_temp(isnan(Similarity_temp)) = 0.5;
    Similarity = ones(num_test,num_BRKNN_neighbour) - Similarity_temp;
    sum_Similarity = sum(Similarity,2);
    Weight = bsxfun(@rdivide,Similarity,sum_Similarity);
    Outputs(i,:) = sum(Weight .* K_neighbour_target,2)';


    Pre_Labels(i,:) = Outputs(i,:);
    Pre_Labels(i,(Pre_Labels(i,:)>=0.5)) = 1;
    Pre_Labels(i,(Pre_Labels(i,:)<0.5)) = -1;
end


