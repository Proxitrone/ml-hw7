function [W_opt, means] = myLDA(classes_data, q, reg_data, W_pca)
%MYLDA Perform LDA on Yale dataset
%   Compute class centers first
    classes = size(classes_data, 1);
    % Compute class means
    means = zeros(size(classes_data{1},1),classes);
    for c=1:classes
        means(:, c) = mean(classes_data{c}, 2);
    end

    % Show class means
    figure('Name','LDA: class means');
    for c=1:classes
        A = reshape(means(:, c), [41, 29]);
        subplot(3, 5, c);
        imshow(A);
    end
    
    % Compute all data mean
    data_mean = mean(means, 2);
    reg_data = reg_data-data_mean;
    % Compute between-class scatter
    S_b = 0;
    for c=1:classes
        class_elem_num = size(classes_data{c},2);
        diff = means(:,c)-data_mean;
        S_b = S_b + class_elem_num*(diff*diff');
    end
    
    % Compute within-class scatter
    S_w = 0;
    for c=1:classes
        class_elem_num = size(classes_data{c},2);
        for i=1:class_elem_num
            diff = (classes_data{c}(:,i)-means(:,c))-means(:,c);
            S_w = S_w + (diff*diff');
        end
    end
    % Get the largest eigenvalues
    W_pca = W_pca(:,1:q);
    
    S_bb = W_pca'*S_b*W_pca;
    S_ww = W_pca'*S_w*W_pca;
    % Compute S_w^-1*S_b and take first q largest eigenvectors as W
    Tmp = S_ww\S_bb;
    
    [eigVec, eigVal] = eig(Tmp);
    % sort eigenvectors and corresponding eigenvalues
    [d, ind] = sort(diag(real(eigVal)), 'ascend');
    eigVal = eigVal(:, ind);
    eigVec = real(eigVec(:, ind));
    
%     W_fld = eigVec(:,1:q);
    W_fld = eigVec;
    % Fisherfaces
    W_opt = W_fld'*W_pca';
    W_opt = W_opt';
    % Display the fisherfaces
    figure('Name', 'LDA: fisherfaces');
    for i=1:size(W_opt, 2)
        A = reshape(W_opt(:, i), [41, 29]);
        subplot(5, 5, i);
        %A = abs(A)./max(abs(A));
        A = A+abs(min(A, [], 'all'));
        A = A./max(A, [], 'all');
        imshow(A);
    end
    
    % Get data after projection
    z = reg_data'*W_opt;
    
    reconstructed = W_opt*z';
    % Show 10 random reconstructed
    figure('Name', 'LDA: reconstructed images');
    for i=1:10
        k=uint8(rand*134+1);
        A = reshape(reconstructed(:, k), [41, 29]);
        subplot(2, 5, i);
        A = A+abs(min(A, [], 'all'));
        A = A./max(A, [], 'all');
        imshow(A);
    end
end

