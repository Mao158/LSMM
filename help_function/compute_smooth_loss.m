function [obj, delta] = compute_smooth_loss(dist_vector)
% Compute smooth hinge loss and delta from dis_vector

obj = zeros(size(dist_vector));
delta = zeros(size(dist_vector));

idx_gt_1 = dist_vector > 1;
idx_lt_0 = dist_vector < 0;
idx_else = ~(idx_gt_1 | idx_lt_0);

obj(idx_gt_1) = 0;
delta(idx_gt_1) = 0;

obj(idx_lt_0) = 0.5 - dist_vector(idx_lt_0);
delta(idx_lt_0) = 1;

obj(idx_else) = 0.5 * (dist_vector(idx_else) - 1).^2;
delta(idx_else) = 1 - dist_vector(idx_else);

% num_T = size(dist_vector,1);
% obj_temp = 0;
% delta = zeros(num_T,1);

% for ii = 1:num_T
%     if dist_vector(ii) > 1
%         obj_temp = obj_temp + 0;
%         delta(ii,1) = 0;
%     elseif dist_vector(ii) < 0
%         obj_temp = obj_temp + 0.5 - dist_vector(ii);
%         delta(ii,1) = 1;
%     else
%         obj_temp = obj_temp + 0.5 * (dist_vector(ii)-1) * (dist_vector(ii)-1);
%         delta(ii,1) = 1 - dist_vector(ii);
%     end
% 
% end
end

