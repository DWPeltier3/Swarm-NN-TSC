classdef ConverterForPatchEmbeddingLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2023 The MathWorks, Inc.

    % DLT formats (labels in [] are optional):
    % Input:  S...C[BTU]
    % Output: SC[BTU]
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            patchSize = this.Layer.PatchSize;
            weights = this.Layer.Weights;
            bias = this.Layer.Bias;
            layerName = this.OutputTensorName + "_";
            biasShape = numel(bias);
            if strcmp(this.Layer.SpatialFlattenMode, 'column-major')
                flip = 'True';
            else
                flip = 'False';
            end
            
            if this.layerAnalyzer.IsTemporal
                % Make time dimension a dummy spatial dimension
                temporal = 'True';
                patchSize = [1, patchSize];
                ndimsWeights = ndims(weights);
                weights = permute(weights, [ndimsWeights+1, 1:ndimsWeights]);
            else
                temporal = 'False';
            end
            patchSizeStr = sprintf("[%s]", join(string(patchSize), ','));
            weightsShape = size(weights);
            weightsShapeStr = sprintf("[%s]", join(string(weightsShape), ','));

            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "PatchEmbeddingLayer", "patch_size=%s, kernel_shape=%s, bias_shape=%d, flip=%s, temporal=%s", {patchSizeStr, weightsShapeStr, biasShape, flip, temporal}, ...
                layerName, false);
            convertedLayer.customLayerNames = "PatchEmbeddingLayer";

            % Switch memory ordering of weights to row-major for TF
            tfWeights = permute(weights, ndims(weights):-1:1);
            convertedLayer.weightNames = ["kernel", "bias"];
            convertedLayer.weightArrays = {tfWeights, bias};
            convertedLayer.weightShapes = {weightsShape, biasShape};
            convertedLayer.LayerName = layerName;
        end
    end
end