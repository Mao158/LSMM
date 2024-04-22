function triplets = generate_knntriplets(data, target, num_positive, num_negative)

is_nearest = 1;
is_farthest = 0;

num_data = size(data,1);
triplets = zeros(num_data * num_positive * num_negative, 3);
class = unique(target);
diff_index = zeros(num_negative, num_data);

for cc = 1:length(class)
    i = find(target == class(cc));
    j = find(target ~= class(cc));
    class_size_neg = size(j,1);
    nn = LSKnn(data(j,:), data(i,:), 1:num_negative, class_size_neg, is_nearest);
    
    % if the size of negative class is too small to find enough negatives
    % we keep the elements to zero and delete them later
    diff_index(1:min(num_negative, class_size_neg), i) = j(nn(1:min(num_negative, class_size_neg), :));
end

same_index = zeros(num_positive, num_data);
for cc = 1:length(class)
    i = find(target == class(cc));
    class_size_pos = size(i,1);
    nn = LSKnn(data(i,:),data(i,:), 2:num_positive + 1, class_size_pos, is_nearest);

    % if the size of positive clas too small to find enough positives
    % we keep the elements to zero and delete them later
    same_index(1:min(num_positive, class_size_pos - 1), i) = i(nn(1:min(num_positive, class_size_pos - 1), :));
end

clear i j nn;
triplets(:, 1) = vec(repmat(1:num_data, num_positive * num_negative, 1));
% temp = zeros(num_positive * num_negative, num_data);
% for i = 1:num_positive
%     temp((i - 1) * num_negative + 1 : i * num_negative, :) = repmat(same_index(i, :), num_negative, 1);
% end

temp = repmat(same_index, num_negative, 1);
row_indices = kron(1:num_positive, ones(1, num_negative));
temp = temp(row_indices, :);
triplets(:, 2) = vec(temp);
clear row_indices temp;

triplets(:, 3) = vec(repmat(diff_index, num_positive, 1));

% remove missing triplets (0 valued)
clear same_index diff_index;
triplets = triplets(all(triplets, 2), :);


