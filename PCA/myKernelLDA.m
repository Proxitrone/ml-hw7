function [W_opt, means] = myKernelLDA(data, k, kernel_type)
%MYKERNELLDA Perform Kernel LDA
%   Compute within-class and between-class scatter using a kernel
% Need to center data in kernel space

    
    obs_num = size(data, 2);
    class_num = 15;
    % Generate class vectors
    class_vec = zeros(obs_num, class_num);
    for c=1:class_num
        sample_num = 9;
        for i=1:sample_num
            class_vec((c-1)*sample_num + i, c) = 1;
        end
    end

    % Compute the gram matrix
    K = compute_kernel(data, data, kernel_type);
    K = K./obs_num;

    %Compute dual representation of averages (in kernel space)
    means = zeros(obs_num, class_num);
    data_mean = zeros(obs_num, 1);

    class_obs = 11;
    for c=1:class_num
        means(:, c) = K* class_vec(:, c) ./class_obs;
        data_mean = data_mean + means(:, c);
    end
    data_mean = data_mean./class_num;

    % Compute the within-class and between-class scatter
    M = zeros(obs_num, obs_num);
    N = K*K';

    for c = 1:class_num
        M = M+(means(:, c)-data_mean)*(means(:, c)-data_mean)'; 
        N = N-(sample_num*(means(:, c)*means(:, c)'));
    end
    M= M.*(class_num-1); %between-class unbiased scatter
   
   %Regularizing 
    mK = abs(mean(K(:)));
   
    C = 0.25*mK;
   
    N = N+C*K;
   
   % Extract eigenvalues and eigenvectors
    [Vtmp, lambda] = eig(M, N);
    lambda = real(diag(lambda));
    
    % Sort the eigenvalues.
    [~, index] = sort(abs(lambda), 'descend');
    W_opt = Vtmp(:, index);
    
    z = K*W_opt;
%     figure;
%     hold on
%     for c = 1 : class_num
%         zC = z(logical(class_vec(:, c)),1:3);
%         plot3(zC(:,1), zC(:,2), zC(:,3), 'x')
%     end
%     xlabel('DA 1')
%     ylabel('DA 2')
%     zlabel('DA 3')
%     title('Projected points from the training data')
%     hold off
end

