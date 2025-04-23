classdef ConverterForEmbeddingConcatenationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2023 The MathWorks, Inc.

    % DLT input must have a 'C' dimension and one 'S' or a T' dimension.
    
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            layerName = this.OutputTensorName + "_";
            weights = this.Layer.Weights;
            weightsShape = [1, numel(weights)];
            weightsShapeStr = sprintf("[%s]", join(string(weightsShape), ','));
            
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "EmbeddingConcatenationLayer", "kernel_shape=%s", {weightsShapeStr}, ...
                layerName, false);
            convertedLayer.customLayerNames = "EmbeddingConcatenationLayer";

            convertedLayer.weightNames = "kernel";
            convertedLayer.weightArrays = {weights};
            convertedLayer.weightShapes = {weightsShape};
            convertedLayer.LayerName = layerName;
        end
    end
end