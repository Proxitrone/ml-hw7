function [Gram] = compute_kernel(dataA, dataB, kernel_type)
%COMPUTE_KERNEL Compute the gram matrix
%   Different kernels available: rbf, gauss, 2nd order polynomial
    gamma_s = 1/100;
    
    K = pdist2(dataA', dataB', 'euclidean');
    
    if strcmp(kernel_type, 'rbf')
        Gram = exp(-gamma_s*K);
    end
    
    if strcmp(kernel_type, 'gauss')
        Gram = (gamma_s/pi)*exp(-gamma_s*K);
    end
    
    if strcmp(kernel_type, 'poly')
        Gram = (dataA'*dataB+gamma_s).^2;
    end
    
end

