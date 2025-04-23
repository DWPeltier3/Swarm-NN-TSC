function ImageInputSize = validateImageInputSize(ImageInputSize)
    % Copyright 2021 The MathWorks, Inc.
    if ~isempty(ImageInputSize) && ~(isreal(ImageInputSize) && isvector(ImageInputSize) && ...
            ismember(numel(ImageInputSize),[2 3 4]) && ...
            isequal(ImageInputSize, floor(ImageInputSize)) && ...
            all(ImageInputSize > 0))
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ImageInputSizeBad')));
    end
    ImageInputSize = double(ImageInputSize(:));
end