function LayersOrGraph = configureOutputLayer(LayersOrGraph, OutputLayerType, placeholderindices)  
    % Copyright 2021 The MathWorks, Inc.    
    isOutputSizeValid = isempty(placeholderindices);
    na = nnet.internal.cnn.analyzer.NetworkAnalyzer(LayersOrGraph);
    ClassificationOutputLayerIndices = find([na.LayerAnalyzers.IsClassificationLayer]);
    CVSTInstalled = nnet.internal.cnn.keras.isInstalledCVST;
    
    for i = ClassificationOutputLayerIndices
        % loop through each classification output layer
        naOut = na.LayerAnalyzers(i); 
        newOutputLayer = [];
        isPixelClassification = false;
        if isOutputSizeValid
            outputTensorSize = naOut.Inputs{'in', 'Size'}{1}; 
            isPixelClassification = numel(outputTensorSize) == 3 && ~isequal(outputTensorSize(1:2), [1 1]);
            if isPixelClassification && CVSTInstalled 
                newOutputLayer = pixelClassificationLayer('Name', naOut.Name); 
            end
        else
            if strcmpi(OutputLayerType, 'PixelClassification') && CVSTInstalled
                newOutputLayer = pixelClassificationLayer('Name', naOut.Name); 
            end
        end
        
        % Remove or update output layer if applicable 
        if ~CVSTInstalled && isPixelClassification
            % CVST is not installed. However, a pixel-classification layer
            % is supposed to exist. So we remove the incorrect
            % classification layer
            nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:PixelClassificationLayerOmitted'); 
            if isa(LayersOrGraph, 'nnet.cnn.LayerGraph')
                LayersOrGraph = removeLayers(LayersOrGraph, naOut.Name); 
            else
                LayersOrGraph(i) = [];
            end
        elseif ~isempty(newOutputLayer) 
            % Replace the classification output layer 
            if isa(LayersOrGraph, 'nnet.cnn.LayerGraph')
                LayersOrGraph = replaceLayer(LayersOrGraph, naOut.Name, newOutputLayer); 
            else
                LayersOrGraph(i) = newOutputLayer; 
            end
        end
    end
end