function [W_pca] = myPCA(data, k)
%MYPCA Use PCA to show the first 25 eigenfaces and reconstruction
%   Compute the Covariance matrix and take the first k eigenvectors
    data_mean = mean(data, 2);
    data = data-data_mean;
    C = cov(data');
    [eigVec, eigVal] = eig(C);
    % sort eigenvectors and corresponding eigenvalues
    [d, ind] = sort(diag(eigVal), 'descend');
    eigVal = eigVal(:, ind);
    eigVec = eigVec(:, ind);
    
    W = eigVec(:,1:k);
    W_pca = eigVec;
    %Get data after projection
    z=data'*W;
    
    % Display the eigenfaces
    figure('Name', 'PCA: eigenfaces');
    for i=1:k
        A = reshape(W(:, i), [41, 29]);
        subplot(k/5, 5, i);
        %A = abs(A)./max(abs(A));
        A = A+abs(min(A, [], 'all'));
        A = A./max(A, [], 'all');
        imshow(A);
    end
    % Build the reconstructed images
    reconstructed = W*z';
    % Show 10 random reconstructed
    figure('Name', 'PCA: reconstructed images');
    for i=1:10
        k=uint8(rand*134+1);
        A = reshape(reconstructed(:, k), [41, 29]);
        subplot(2, 5, i);
        A = A+abs(min(A, [], 'all'));
        A = A./max(A, [], 'all');
        imshow(A);
    end
end

