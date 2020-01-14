function [Train,Test] = load_yale(root, crop_param, target_rescale)
%LOAD_YALE Load the Yale_Face_Database into matrices in matlab
%   Train and Test sets are cells with matrices for each test_subject (class)
%   Perform normalization and resizing 
    train_subj_num = 13;
    test_subj_num = 2;
    subj_express_num = 11;
    face_vec_dim = target_rescale(1)*target_rescale(2);
    
    train_path = [root ,'Yale_Face_Database\Training\'];
    test_path = [root ,'Yale_Face_Database\Testing\'];
    
    D_train = dir(fullfile(train_path, '*.pgm'));
    D_test = dir(fullfile(test_path, '*.pgm'));
    
    
    Train = zeros(face_vec_dim, numel(D_train));
    Test = zeros(face_vec_dim, numel(D_test));
    
    %% Load the train set
    for k=1:numel(D_train)
       face = imread(fullfile(train_path, D_train(k).name));
       face = preprocess_face(face, crop_param, target_rescale);
       Train(:, k) = face(:);
    end
    
    %% Load the test set
    for k=1:numel(D_test)
       face = imread(fullfile(test_path, D_test(k).name));
       face = preprocess_face(face, crop_param, target_rescale);
       Test(:, k) = face(:);
    end
    
    
end

