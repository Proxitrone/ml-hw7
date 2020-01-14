close all;
% clear all;
clc;

rng(10);
% Load Yale database
root = [];
target_rescale = [41, 29];
crop_param = [47, 60, 100, 150];
[Train, Test] = load_yale(root, crop_param, target_rescale);

k = 25;
W_pca=0;
W_pca = myPCA(Train, k);

% Divide Training set by classes (every 9 datapoints belong to 1 subject)
classes = 15;
Train_classes = cell(classes, 1);

for c=1:classes
    Train_classes{c} = zeros(size(Train,1), 9);
    for i=1:9
        Train_classes{c}(:,i) = Train(:, (c-1)*9+i);
    end
end

W_lda = myLDA(Train_classes, 10, Train, W_pca);

