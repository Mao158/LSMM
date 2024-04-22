function NN = LSKnn(neg_data, pos_data, ks, class_size, key)
B = 750;
num_pos = size(pos_data,1);
NN = zeros(length(ks),num_pos);

for i = 1:B:num_pos
    BB = min(B,num_pos - i);
    %   fprintf('.');
%     Dist = distance(X1,X2(:,i:i+BB));
    Dist = pdist2(neg_data,pos_data(i:i+BB,:));
    %   fprintf('.');
    %   fprintf('.');
    %     if key == 1 % compute ks nearest points
    %         [dist,nn]=mink(Dist,max(ks));
    %     elseif key == 0 % compute ks farthest points
    %         [dist,nn]=maxk(Dist,max(ks));
    %     end
    if key == 1 % compute ks nearest points
        [~, idex] = sort(Dist,1,'ascend');
    elseif key == 0 % compute ks farthest points
        [~, idex] = sort(Dist,1,'descend');
    end
    clear('Dist');

    if class_size < max(ks)
        idex = [idex; zeros((max(ks) - class_size),size(idex,2))];
    end
    nn = idex(ks, :);
    NN(:,i:i+BB) = nn;
    clear('nn');
end

end

