function [W_pca] = myKernelPCA(data, k, kernel_type)
%MYKERNELPCA Perform Kernel PCA
%   Same as PCA, but now instead of covariance matrix we use kernel matrix
    
    data_mean = mean(data, 2);
    data = data-data_mean;
    C = compute_kernel(data, data, kernel_type);
    [eigVec, eigVal] = eig(C);
    % sort eigenvectors and corresponding eigenvalues
    [d, ind] = sort(diag(eigVal), 'descend');
    eigVal = eigVal(:, ind);
    eigVec = eigVec(:, ind);
    
    W = eigVec(:,1:k);
    W_pca = eigVec;
    %Get data after projection
    z=data*W;
    
    % Display the eigenfaces
%     figure('Name', 'Kernel PCA: eigenfaces');
%     for i=1:k
%         A = reshape(W(:, i), [41, 29]);
%         subplot(k/5, 5, i);
%         %A = abs(A)./max(abs(A));
%         A = A+abs(min(A, [], 'all'));
%         A = A./max(A, [], 'all');
%         imshow(A);
%     end
%     % Build the reconstructed images
%     reconstructed = W*z';
%     % Show 10 random reconstructed
%     figure('Name', 'Kernel PCA: reconstructed images');
%     for i=1:10
%         k=uint8(rand*134+1);
%         A = reshape(reconstructed(:, k), [41, 29]);
%         subplot(2, 5, i);
%         A = A+abs(min(A, [], 'all'));
%         A = A./max(A, [], 'all');
%         imshow(A);
%     end
end

