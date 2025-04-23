function OutputLayerType = validateKerasOutputLayerType(OutputLayerType)
    % Copyright 2021 The MathWorks, Inc.
    if ~(isa(OutputLayerType,'char') || isa(OutputLayerType,'string'))
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:OutputLayerTypeString')));
    end
    OutputLayerType = lower(char(OutputLayerType)); 
    supportedOutputTypes = {'classification', 'pixelclassification', 'regression', 'binarycrossentropyregression'}; 
    if ~(isempty(OutputLayerType) || ismember(OutputLayerType, supportedOutputTypes))
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedOutputLayerType'))); 
    end 
    if isequal(OutputLayerType,'pixelclassification') && ~nnet.internal.cnn.keras.isInstalledCVST
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:noCVSTForPixelClassification')));
    end
end