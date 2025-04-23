function s = getKerasFileExtension(filename)
    % Copyright 2021 The MathWorks, Inc.
    [~,~,ext] = fileparts(filename);
    if isempty(ext)
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnknownFormat');
        s = '.h5';
    else
        s = ext;
    end
end