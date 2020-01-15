close all;
% clear all;
clc;

rng(10);
% Load Yale database
root = [];
target_rescale = [41, 29];
crop_param = [47, 60, 100, 150];
[Train, Test] = load_yale(root, crop_param, target_rescale);

%% Divide Training and Testing set by classes (every 9 datapoints belong to 1 subject)
classes = 15;
Train_classes = cell(classes, 1);
Test_classes = cell(classes, 1);
for c=1:classes
    Train_classes{c} = zeros(size(Train,1), 9);
    for i=1:9
        Train_classes{c}(:,i) = Train(:, (c-1)*9+i);
    end

    Test_classes{c} = zeros(size(Test,1), 2);
    for i=1:2
        Test_classes{c}(:,i) = Test(:, (c-1)*2+i);
    end
end

k = 25;
W_pca=0;
%% Perform PCA
W_pca = myPCA(Train, k);

%% Perform LDA
[W_lda, class_means] = myLDA(Train_classes, 10, Train, W_pca);
%% Classify regular PCA and LDA
[acc_pca, acc_lda] = classify_pca_lda(W_pca, W_lda, Test_classes, class_means);
disp(['Pca acc: ', num2str(acc_pca)]);
disp(['Lda acc: ', num2str(acc_lda)]);
%% Perform Gauss Kernel PCA
W_gauss_pca = myKernelPCA(Train, k, 'poly');

%% Perform Gauss Kernel LDA
[W_gauss_lda, class_means] = myKernelLDA(Train, k, 'poly');

%% Classify Gauss kernel PCA and LDA
[acc_pca, acc_lda] = classify_kernel_pca_lda(W_gauss_pca, W_gauss_lda, Test_classes, class_means, Train, Test, 'poly');
disp(['poly pca acc: ', num2str(acc_pca)]);
disp(['poly lda acc: ', num2str(acc_lda)]);
%% Perform RBF Kernel PCA
W_rbf_pca = myKernelPCA(Train, k, 'rbf');

%% Perform RBF Kernel LDA
[W_rbf_lda, class_means] = myKernelLDA(Train, k, 'rbf');

%% Classify RBF kernel PCA and LDA
[acc_pca, acc_lda] = classify_kernel_pca_lda(W_rbf_pca, W_rbf_lda, Test_classes, class_means, Train, Test, 'rbf');
disp(['RBF pca acc: ', num2str(acc_pca)]);
disp(['RBF lda acc: ', num2str(acc_lda)]);









