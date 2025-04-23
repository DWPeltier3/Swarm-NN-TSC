function NNTInputShape = determineImageInputSize(KerasInputShape, UserImageInputSizeArg)
% Determine the image input size given the Keras input shape and the passed
% argument ImageInputSize. KerasInputShape can be [NaN h w] or [NaN h w c] 
% or [NaN h w d c]. UserImageInputSizeArg can be [h w] or [h w c] or 
% [h w d c]. NNTInputShape can be [h w c] or [h w d c].

%   Copyright 2017-2020 The MathWorks, Inc.

assert(ismember(numel(KerasInputShape), [3 4 5]));
% Convert KerasInputShape into a 1x3 vector
KerasInputShape = KerasInputShape(:)';
KerasInputShape = KerasInputShape(2:end);
if numel(KerasInputShape) == 2
    KerasInputShape(3) = 1;     % 1 channel if none specified.
end
NNTInputShape = KerasInputShape;
if ~isempty(UserImageInputSizeArg)
    % Convert UserImageInputSizeArg into a 1x3 vector if input is 2D
    % Convert UserImageInputSizeArg into a 1x4 vector if input is 3D
    UserImageInputSizeArg = UserImageInputSizeArg(:)';
    if numel(UserImageInputSizeArg) == 2
        UserImageInputSizeArg(3:numel(KerasInputShape)) = KerasInputShape(3:numel(KerasInputShape)); 
        % Fill in Keras num channels if none specified. This is the
        % scenarios if UserImageInputSizeArg is [h w]
    end
    for element = 1:numel(UserImageInputSizeArg)   % h,w,c or h,w,d,c
        if isnan(KerasInputShape(element))
            NNTInputShape(element) = UserImageInputSizeArg(element);
        elseif KerasInputShape(element) ~= UserImageInputSizeArg(element)
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ImageInputSizeMismatch',...
                element, num2str(UserImageInputSizeArg(element)), num2str(KerasInputShape(element)))));
        end
    end
end
end
