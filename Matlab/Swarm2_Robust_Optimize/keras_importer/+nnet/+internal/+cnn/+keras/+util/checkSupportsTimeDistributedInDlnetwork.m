function tf = checkSupportsTimeDistributedInDlnetwork(layerConfig)
    % checkSupportsTimeDistributedInDlnetwork layer must be 
    % compatible with spatio-temporal inputs and the
    % import call must be from the TensorFlow importer

    % Copyright 2023 The MathWorks, Inc.

    % List of layers that can support the extra time dimension
    % label to bypass wrapping inside sequenceFolding/ unFolding layers
    supportedLayers = {'InputLayer', 'Activation', 'PRELU', 'ReLU', 'LeakyReLU', 'ELU', 'Softmax', ...
        'Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'Conv3DTranspose' 'AveragePooling1D', 'AveragePooling2D', 'AveragePooling3D', ...
        'GlobalAveragePooling1D','GlobalAveragePooling2D', 'GlobalAveragePooling3D', 'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D', ...
        'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'GlobalMaxPooling3D', 'LSTM', 'GRU', 'Dense'};

    % The layer inside the TimeDistributed wrapper can also be a
    % sequential model. Check if the layers inside the sequential
    % model all support the extra time dimension label
    if strcmp(layerConfig.class_name, 'Sequential')
        tf = all(arrayfun(@(layer) ismember(layer.class_name, supportedLayers), layerConfig.config.layers));
    else
        tf = ismember(layerConfig.class_name, supportedLayers);
    end
end