function [obj,grad] = compute_Se_grad_L(L, train_data, side_info, para)
% Compute gradient and objective value given current solution (transformation L)

gamma = para.gamma;
alphfa = para.alphfa;
lambda_1 = para.lambda_1;
lambda_2 = para.lambda_2;

num_data = size(train_data, 1);
num_label = size(side_info, 1);
num_metric = 2 * num_label + 1;

L = mat(L, num_metric);
grad = zeros(size(L));
obj = 0;

%% empirical loss term
for kk = 1:num_label
    num_T = size(side_info{kk}, 1);
    curr_pos_L = mat((L(2*kk-1,:)' + L(end,:)'), size(train_data, 2));
    curr_neg_L = mat((L(2*kk,:)' + L(end,:)'), size(train_data, 2));

    pos_indices = side_info{kk}(:, 3) == 1;
    neg_indices = side_info{kk}(:, 3) == -1;
    pair_distance = zeros(num_T,1);
    pair_distance(pos_indices) = sum(((train_data(side_info{kk}(pos_indices, 1), :) - train_data(side_info{kk}(pos_indices, 2), :)) * curr_pos_L).^2, 2);
    pair_distance(neg_indices) = sum(((train_data(side_info{kk}(neg_indices, 1), :) - train_data(side_info{kk}(neg_indices, 2), :)) * curr_neg_L).^2, 2);

    dist_vector = (gamma - alphfa * pair_distance) .* side_info{kk}(:, 3);
    clear pos_indices neg_indices pair_distance;

    % compute smooth hinge loss and delta from the dist_vector
    [obj_vec, delta_temp] = compute_smooth_loss(dist_vector);
    delta = alphfa * delta_temp;
    pos_ins_indices = find(side_info{kk}(:, 4) == 1);
    neg_ins_indices = find(side_info{kk}(:, 4) == 0);
    num_pos_ins = size(pos_ins_indices, 1);
    num_neg_ins = size(neg_ins_indices, 1);
    obj = obj + (sum(obj_vec(pos_ins_indices)) / num_pos_ins + sum(obj_vec(neg_ins_indices)) / num_neg_ins) / num_label;
    clear delta_temp obj_vec dist_vector

    % compute gradient for local and global metric
    delta_theta = delta .* side_info{kk}(:, 3);

    SS_pos = sparse(side_info{kk}(pos_ins_indices,1), side_info{kk}(pos_ins_indices,2), delta_theta(pos_ins_indices), num_data, num_data);
    SS_neg = sparse(side_info{kk}(neg_ins_indices,1), side_info{kk}(neg_ins_indices,2), delta_theta(neg_ins_indices), num_data, num_data);

    grad_pos = SODW(train_data', SS_pos) * curr_pos_L / num_pos_ins / num_label * 2;
    grad_neg = SODW(train_data', SS_neg) * curr_neg_L / num_neg_ins / num_label * 2;
    clear delta delta_theta neg_ins_indices pos_ins_indices SS_pos SS_neg curr_pos_L curr_neg_L

    grad(2*kk-1, :) = grad(2*kk-1, :) + vec(grad_pos)';
    grad(2*kk, :) = grad(2*kk, :) + vec(grad_neg)';
    grad(end, :) = grad(end, :) + vec(grad_pos)' + vec(grad_neg)';
    clear grad_neg grad_pos

    % compute lambda1 term in objective
    obj = obj + sum(sum(L((2*kk-1):2*kk,:).^2)) * lambda_1 / num_label / 2;

end

%% for regularizers (F-norm on L)
grad(end, :) = grad(end, :) + 2 * lambda_2 * L(end,:);
grad(1:end-1, :) = grad(1:end-1, :) + L(1:end-1,:) * lambda_1 / num_label;
obj = obj + lambda_2 * sum(L(end,:).^2);

grad = vec(grad);
end

