function [acc_pca, acc_lda] = classify_pca_lda(W_pca, W_lda, classes_test_data, means)
%CLASSIFY_PCA_LDA Classify images from test set using pca and lda
%   To classify test images, we need to project each image using
%   corresponding transformation, project class means, compute distance of
%   our test projection to each of the projected means, minimum distance
%   mean is going to be our class
    W_pca = W_pca(:, 1:14);
    
    classes_num = size(classes_test_data, 1);
    proj_means_pca = W_pca'*means;
    proj_means_lda = W_lda'*means;
    acc_pca = 0;
    acc_lda = 0;
    num_data = 0;
    for c=1:classes_num
        class_elem_num = size(classes_test_data{c},2);
        for i=1:class_elem_num
            sample_proj_pca = W_pca'*classes_test_data{c}(:,i);
            sample_proj_lda = W_lda'*classes_test_data{c}(:,i);
            distance_pca = zeros(1, classes_num);
            distance_lda = zeros(1, classes_num);
            for k=1:classes_num
                distance_pca(1, k) = norm(sample_proj_pca-proj_means_pca(:,k))^2;
                distance_lda(1, k) = norm(sample_proj_lda-proj_means_lda(:,k))^2;
            end
            
            [val, ind_pca] = min(distance_pca);
            if ind_pca == c
                acc_pca = acc_pca +1;
            end
            [val, ind_lda] = min(distance_lda);
            if ind_lda == c
                acc_lda = acc_lda +1;
            end
            num_data = num_data+1;
        end
    end
    
    acc_pca = acc_pca/num_data;
    acc_lda = acc_lda/num_data;
    
end

