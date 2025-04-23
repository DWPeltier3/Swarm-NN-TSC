function checkInputSizeForMINetworks(AM, UserImageInputSize, isNetwork)
    % Copyright 2021 The MathWorks, Inc.
    % Check specifically for a multiple input network, and throw an error
    % that this case is not supported if the image input size is missing.
    % Also request the user to switch to importKerasLayers. 
    if strcmp(AM.ModelType, 'DAG') && numel(AM.InputLayerIndices) > 1
        if ~isempty(UserImageInputSize) 
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MIImageInputSize'))); 
        end        
        if isNetwork
            for i = 1:numel(AM.InputLayerIndices)
                curConfig = AM.LayerSpecs{AM.InputLayerIndices(i)};
                curInputSize = kerasField(curConfig, 'batch_input_shape'); 
                if any(isnan(curInputSize(2:end)))
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MINetworkImageInputSizeNeeded'))); 
                end
            end
        end
    end
end