classdef ConverterForPositionEmbeddingLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2023 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            layerName = this.OutputTensorName + "_";
            kerasWeightsShape = [this.Layer.MaxPosition, this.Layer.OutputSize];
            weightsShapeStr = sprintf("[%s]", join(string(kerasWeightsShape), ','));

            % Get TF equivalent of position dimension
            tfPositionDimension = 1;
            if strcmp(this.Layer.PositionDimension, "spatial") && contains(this.InputFormat, "T")
                tfPositionDimension = 2;
            end
            
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "PositionEmbeddingLayer", "axis=%d, params_shape=%s", {tfPositionDimension, weightsShapeStr}, ...
                layerName, false);
            convertedLayer.customLayerNames = "PositionEmbeddingLayer";

            % MATLAB weights is [outputSize, maxPosition]. This should be
            % [maxPosition, outputSize] in TF since the dimension ordering
            % is flipped between MATLAB and TF
            tfWeights = this.Layer.Weights;
            convertedLayer.weightNames = "params";
            convertedLayer.weightArrays = {tfWeights};
            convertedLayer.weightShapes = {kerasWeightsShape};
            convertedLayer.LayerName = layerName;
        end
    end
end