function [obj,grad] = compute_Cl_grad_L(L, train_data, side_info, para)
% Compute gradient and objective value given current solution (transformation L)

gamma = para.gamma;
alphfa = para.alphfa;
lambda_1 = para.lambda_1;
lambda_2 = para.lambda_2;
num_cluster = para.num_cluster;

num_data = size(train_data, 1);
num_label = size(side_info, 1);
num_metric = num_cluster * num_label + 1;

L = mat(L, num_metric);
grad = zeros(size(L));
obj = 0;

%% empirical loss term
for kk = 1:num_label
    num_T = size(side_info{kk}, 1);

    indices_cluster_idx = (num_cluster * (kk - 1) + 1) : (num_cluster * kk);
    cluster_L = arrayfun(@(i) mat((L(indices_cluster_idx(i), :)' + L(end, :)'), size(train_data, 2)), 1:num_cluster, 'UniformOutput', false);

    pair_distance = zeros(num_T, 1);
    for i = 1:num_T
        pair_distance(i) = sum(((train_data(side_info{kk}(i, 1), :) - train_data(side_info{kk}(i, 2), :)) * cluster_L{side_info{kk}(i, 4)}).^2, 2);
    end    

    dist_vector = (gamma - alphfa * pair_distance) .* side_info{kk}(:, 3);
    clear indices_cluster_idx pair_distance;

    % compute smooth hinge loss and delta from the dist_vector
    [obj_vec, delta_temp] = compute_smooth_loss(dist_vector);
    delta = alphfa * delta_temp;
    delta_theta = delta .* side_info{kk}(:, 3);
    clear delta_temp dist_vector delta

    for i = 1:num_cluster
        cluster_ins_indices = find(side_info{kk}(:, 4) == i);
        num_cluster_ins = size(cluster_ins_indices, 1);
        obj = obj + sum(obj_vec(cluster_ins_indices)) / num_cluster_ins / num_label;

        % compute gradient for local and global metric
        SS = sparse(side_info{kk}(cluster_ins_indices,1), side_info{kk}(cluster_ins_indices,2), delta_theta(cluster_ins_indices), num_data, num_data);
        grad_temp = SODW(train_data', SS) * cluster_L{i} / num_cluster_ins / num_label * 2;
        grad(num_cluster * (kk - 1) + i, :) = grad(num_cluster * (kk - 1) + i, :) + vec(grad_temp)';
        grad(end, :) = grad(end, :) + vec(grad_temp)';
    end

    clear obj_vec delta_theta cluster_ins_indices SS cluster_L grad_temp

    % compute lambda1 term in objective
    obj = obj + sum(sum(L((num_cluster * (kk - 1) + 1) : (num_cluster * kk),:).^2)) * lambda_1 / num_cluster / num_label;
end

%% for regularizers (F-norm on L)
grad(end, :) = grad(end, :) + 2 * lambda_2 * L(end,:);
grad(1:end-1, :) = grad(1:end-1, :) +  L(1:end-1,:) * 2 * lambda_1 / num_cluster / num_label;
obj = obj + lambda_2 * sum(L(end,:).^2);

grad = vec(grad);
end

