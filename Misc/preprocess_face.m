function [new_face] = preprocess_face(face, crop_param, target_rescale)
%PREPROCESS_FACE Do some preprocessing on each face before loading
%   Crop images, downsample to 29x41, then normalize and zero-mean
    
    % Crop
    face = imcrop(face, crop_param);
    % Downsample
    face = imresize(face, target_rescale);
    % Normalize
    face = double(face)./256;
    % Zero mean
%     face = (face-mean(face(:)))/std(face(:));
    
    
    new_face = face;
end

