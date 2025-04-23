function Format = getFileFormat(FileName, isNetwork)
    % Copyright 2021 The MathWorks, Inc.
    ext = nnet.internal.cnn.keras.util.getKerasFileExtension(char(FileName));
    switch lower(ext)
        case {'.h5','.hdf5'}
            Format = 'hdf5';
        case '.json'
            Format = 'json';
        case '.pb'
            if isNetwork
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedFormatButProtobufImportNetwork', ext)));
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedFormatButProtobufImportLayers', ext)));
            end
        otherwise
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedFormat', ext)));
    end
end