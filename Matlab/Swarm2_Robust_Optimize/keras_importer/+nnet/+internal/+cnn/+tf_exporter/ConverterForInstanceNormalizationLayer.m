classdef ConverterForInstanceNormalizationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            epsilon = this.Layer.Epsilon;
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "tfa.layers.InstanceNormalization", "axis=-1, epsilon=%f", {epsilon}, ...
                layerName, false);                          % Normalization layers must NOT be time-distributed.

            % create tf weights: All 1D vectors. In TF they are have these
            % names and ordering: gamma, beta
            convertedLayer.packagesNeeded = "tfa";
            convertedLayer.weightNames     = ["gamma", "beta"];
            convertedLayer.weightArrays = {this.Layer.Scale, this.Layer.Offset};
            convertedLayer.weightShapes = {numel(this.Layer.Scale), numel(this.Layer.Offset)};
            convertedLayer.LayerName = layerName;
        end
    end
end