function LayersOrGraph = configureOutputLayerForNetwork(LayersOrGraph, PassedClasses)
    % Copyright 2021 The MathWorks, Inc.
	% Get layers
    isLG = isa(LayersOrGraph, 'nnet.cnn.LayerGraph');
    if isLG
        Layers = LayersOrGraph.Layers;
    else
        Layers = LayersOrGraph;
    end
    
    % Error if any unsupported layers, refer user to importKerasLayers
    if any(arrayfun(@nnet.internal.cnn.keras.util.isUnsupportedLayer, Layers))
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantBuildNetWithUnsupportedLayers')));
    end
    
    % Check for RNN (if there is a SequenceInputLayer it is a RNN) 
    isRNN = any(find(arrayfun(@(L)isa(L,'nnet.cnn.layer.SequenceInputLayer'), Layers)));
    % Find output layer
    locC = find(arrayfun(@(L)isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), Layers));
    locR = find(arrayfun(@(L)isa(L,'nnet.cnn.layer.RegressionOutputLayer') ||...
                             isa(L,'nnet.keras.layer.BinaryCrossEntropyRegressionLayer'), Layers));
    if numel(locC) + numel(locR) == 1 % is SO
        if ~isempty(locC)
            na = nnet.internal.cnn.analyzer.NetworkAnalyzer(LayersOrGraph); 
            naOut = na.LayerAnalyzers(locC); 
            outputTensorSizes = naOut.Inputs{'in', 'Size'}{1}; 
            isPixelClassification = numel(outputTensorSizes) == 3 && ~isequal(outputTensorSizes(1:2), [1 1]);
        else
            outputTensorSizes = []; 
            isPixelClassification = false; 
        end

        nnet.internal.cnn.keras.util.checkOutputLayerTypeAndClasses(~isempty(locR), PassedClasses, isPixelClassification, isRNN, outputTensorSizes);
        if isPixelClassification || ~nnet.internal.cnn.keras.util.isAuto(PassedClasses)
            if isPixelClassification
                newOutputLayer = pixelClassificationLayer('Name', naOut.Name, 'Classes', PassedClasses);
            else
                newOutputLayer = classificationLayer('Name', naOut.Name, 'Classes', PassedClasses);
            end
            % Put the new output layer in
            if isLG
                LayersOrGraph = replaceLayer(LayersOrGraph, newOutputLayer.Name, newOutputLayer);
            else
                LayersOrGraph(locC) = newOutputLayer;
            end
        end
    else
        % The imported network is MO so 'Classes' will be ignored, if specified.
        if ~isequal(PassedClasses, 'auto')
            nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:TFNetworkMOClasses');
        end
    end
end 