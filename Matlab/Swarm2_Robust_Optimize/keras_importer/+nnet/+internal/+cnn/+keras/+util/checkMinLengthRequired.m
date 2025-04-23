function [lGraph,minLengthRequired] = checkMinLengthRequired(layersOrGraph)
    % autoSetMinLength  Determine MinLength for Sequence Input Layer
    
    % Copyright 2021 The MathWorks, Inc.
    minLengthRequired = false;
    hasPlaceholderLayers = ~isempty(findPlaceholderLayers(layersOrGraph));
    if isequal(class(layersOrGraph),'nnet.cnn.layer.Layer')
        inputLayer = layersOrGraph(1);
        lGraph     = layerGraph(layersOrGraph);
        [lGraph, minLengthRequired]    = icheckMinLengthHelper(lGraph,inputLayer, hasPlaceholderLayers);
    elseif isequal(class(layersOrGraph),'nnet.cnn.LayerGraph')
        lGraph = layersOrGraph;
        if length(lGraph.InputNames) == 1
            inputLayer = lGraph.Layers(1);
            [lGraph, minLengthRequired]    = icheckMinLengthHelper(lGraph,inputLayer, hasPlaceholderLayers);
        end
    end
end

function [lGraph, minLengthRequired] = icheckMinLengthHelper(lGraph,inputLayer, hasPlaceholderLayers)
    minLengthRequired = false;
    if (any(arrayfun(@(ls)ismember(class(ls),{'nnet.cnn.layer.Convolution1DLayer',...
                'nnet.cnn.layer.MaxPooling1DLayer','nnet.cnn.layer.AveragePooling1DLayer'}), lGraph.Layers)) && ...
                    isequal(class(inputLayer), 'nnet.cnn.layer.SequenceInputLayer'))
            if(inputLayer.MinLength == 1)
                if ~hasPlaceholderLayers
                    minLengthRequired = true;
                else
                    %If LayerGraph has placeholder layers, warn the user about
                    %min length being not automatically set
                    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:MinLengthNotSet',...
                        inputLayer.Name);
                end
            end
    end
end