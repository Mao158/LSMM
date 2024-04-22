function [L,obj_val] = Se_GD(L, train_data, side_info, para, fold)
% gradient descent optimization

% if gpuDeviceCount>0
%     fprintf('GPU detected. Trying to use it ...\n');
%     try
%         L=gpuArray(L);
%         train_data=gpuArray(train_data);
%         side_info=cellfun(@gpuArray, single(side_info), 'UniformOutput', false);
%         fprintf('Using GPU ...\n');
%     catch
%     end
% end
data_str = para.data_str;
num_fold = para.num_fold;
verbose = para.verbose;
max_iter = para.max_iter;
learn_rate = para.learn_rate;

obj_val = zeros(1, max_iter);
[obj, grad] = compute_Se_grad_L(L, train_data, side_info, para);


for iter = 1:max_iter
    while true
        L_next = L - learn_rate .* grad;
        [obj_next, grad_next] = compute_Se_grad_L(L_next, train_data, side_info, para);
        delta_obj = obj_next - obj;

        if delta_obj > 0
            learn_rate = learn_rate / 2;
        else
            break;
        end
    end
    L = L_next;
    grad = grad_next;
    obj = obj_next;
    learn_rate = learn_rate * 1.01;

    obj_val(iter) = obj;

    if verbose
        fprintf('Dataset:%s | Fold:%2d / %d | Iter:%3d | Obj:%10.4f | Obj_diff:%10.4f | Lr:%7.4f\n', data_str, fold, num_fold, iter, obj, delta_obj, learn_rate);
    end

    if iter > 50 && abs(delta_obj) < 0.001
        if verbose
            fprintf('LSMM converged at %d-th iteration with objective %f\n', iter, obj);
        end
        break;
    elseif iter == max_iter && abs(delta_obj) >= 0.001
        if verbose
            fprintf('LSMM did not converge in %d steps\n', iter);
        end
    end

end


end

