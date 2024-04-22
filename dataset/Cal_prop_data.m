clc;
clear;

% load('CAL500.mat');%502,68,174
% load('emotions.mat');%593,72,6
% load('birds.mat');%645,258,19
% load('genbase.mat');%662,1186,27
% load('medical.mat');%978,1449,45
% load('enron.mat');%1702,1001,53
% load('image.mat');%2000,294,5
% load('scene.mat');%2407,294,6
% load('yeast.mat');%2417,103,14
% load('slashdot.mat');%3781,,1079,22
% load('arts.mat');%5000,462,26
% load('corel5k.mat');%5000,499,374
% load('education.mat');%5000,550,33
% load('rcv1subset1_top944.mat');%6000,944,101*
% load('rcv1subset2_top944.mat');%6000,944,101*
% load('bibtex.mat');%7395,1836,159*
% load('business.mat');%8718,581,30
% load('corel16k001.mat');%13766,500,153*
% load('delicious.mat');%16105,500,983*
% load('eurlex_dc.mat');%19348,5000,412*
% load('eurlex_sm.mat');%19348,5000,201*
% load('mirflickr.mat');%25000,1000,38
load('tmc2007.mat');%28596,500,22
% load('mediamill.mat');%43907,120,101
[num_data, num_dim, num_label, Label_cardinality, Label_density, Label_diversity, Proportion_Label_diversity] = cal_Lcard(data, target);




function [num_data, num_dim, num_label, Label_cardinality, Label_density, Label_diversity, Proportion_Label_diversity] = cal_Lcard(data, target)
%Calculate the properties of multi-label datasets, including 
%     num_data                    -  the number of data
%     num_dim                     -  the dimension of data
%     num_label                   -  the number of label
%     Label_cardinality           -  the average number of labels per example
%     Label_density               -  the normalization of label cardinality by the number of possible labels in the label space
%     Label_diversity             -  the number of distinct label sets appeared in the data set
%     Proportion_Label_diversity  -  the normalization of label diversity by the number of examples


[num_data, num_dim] = size(data);
num_label = size(target, 1);
Label_cardinality = sum(sum(target)) / num_data;
Label_density = Label_cardinality / num_label;

num_label_diversity = 1;
diversity_label_set = target(:, 1);
for idex_data = 2 : num_data
    for idex_diversity_set = 1 : num_label_diversity
        judge = (diversity_label_set(:, idex_diversity_set) == target(:, idex_data));
        if all(judge)
            break;
        else
            if idex_diversity_set == num_label_diversity
                diversity_label_set = [diversity_label_set, target(:, idex_data)];
                num_label_diversity = num_label_diversity + 1;
            end
        end
    end
end
Label_diversity = num_label_diversity;

Proportion_Label_diversity = Label_diversity / num_data;

end


