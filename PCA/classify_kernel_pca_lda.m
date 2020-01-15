function [acc_pca, acc_lda] = classify_kernel_pca_lda(W_pca, W_lda, classes_test_data, means, data, test_data, kernel_type)
%CLASSIFY_KERNEL_PCA_LDA Perform classification using kernel PCA and LDA
%   Compute kernel distance between testpoints and trainpoints in feature
%   space, shortest distance is our class

    classes_num = size(classes_test_data, 1);
    num_train_data = size(data, 2);
    num_test_data = size(test_data, 2);
    
    acc_pca = 0;
    acc_lda = 0;
    %Compute distances between points and means in feature space
    W_pca = W_pca(:, 1:14);
    W_lda = W_lda(:, 1:14);
    
    % Compute the test kernel matrix 
    K1 = compute_kernel(data, data, kernel_type);
    K2 = compute_kernel(data, test_data, kernel_type);

    K2 = K2./num_train_data;
    K1 = K1./num_train_data;
    % Project data onto the discriminant axes
    proj_test_pca = K2'*W_pca;
    proj_test_lda = K2'*W_lda;
    
    proj_train_pca = K1'*W_pca;
    proj_train_lda = K1'*W_lda;
    % Compute distances in the projected space
    distances_pca = pdist2(proj_test_pca, proj_train_pca, 'euclidean');
    distances_lda = pdist2(proj_test_lda, proj_train_lda, 'euclidean');
    
    % Find the elements with shortest distances
    [shortest, ind_pca] = min(distances_pca, [], 2);
    [shortest, ind_lda] = min(distances_lda, [], 2);
    
    % Generate class vectors
    class_vec_train = zeros(num_train_data, classes_num);
    class_vec_test = zeros(num_test_data, classes_num);
    for c=1:classes_num
        sample_num_train = 9;
        for i=1:sample_num_train
            class_vec_train((c-1)*sample_num_train + i, c) = 1;
        end
        
        sample_num_test = 2;
        for i=1:sample_num_test
            class_vec_test((c-1)*sample_num_test + i, c) = 1;
        end
    end
    
    % Classify test data point according to their closest neighbor
    for i=1:num_test_data
        if sum(class_vec_test(i, :) == class_vec_train(ind_pca(i), :)) == classes_num
%         [val, target_class] = max(class_vec_train(ind_pca(i), :));
            acc_pca = acc_pca+1;
        end
        
        if sum(class_vec_test(i, :) == class_vec_train(ind_lda(i), :)) == classes_num
%         [val, target_class] = max(class_vec_train(ind_pca(i), :));
            acc_lda = acc_lda+1;
        end
    end
    
    acc_pca = acc_pca/num_test_data;
    acc_lda = acc_lda/num_test_data;
end

