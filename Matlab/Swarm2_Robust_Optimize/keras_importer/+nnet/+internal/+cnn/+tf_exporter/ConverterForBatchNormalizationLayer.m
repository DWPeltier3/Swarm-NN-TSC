classdef ConverterForBatchNormalizationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]. In TF, all of these tensors end
    % in C (see table in base class). Since the axis to compute averages
    % for is always C, the TF axis number is always -1. That's the default,
    % so we don't specify it in the generated code.
    methods
        function convertedLayer = toTensorflow(this)
            Epsilon = this.Layer.Epsilon;
            % create tf weights: All 1D vectors. In TF they have these
            % names and ordering: gamma, beta, moving_mean,
            % moving_variance. Batchnorm averages-out all dimensions except
            % C. E.g., if X has SCB format, the mean would be computed as
            % mean(X,[1,3]).
            gamma = this.Layer.Scale;
            beta = this.Layer.Offset;
            moving_mean = this.Layer.TrainedMean;
            moving_variance = this.Layer.TrainedVariance;

            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "layers.BatchNormalization", "epsilon=%f", {Epsilon}, ...
                layerName, false);                          % Normalization layers must NOT be time-distributed.
            convertedLayer.weightNames = ["gamma", "beta", "moving_mean", "moving_variance"];
            convertedLayer.weightArrays = {gamma, beta, moving_mean, moving_variance};
            convertedLayer.weightShapes = {numel(gamma), numel(beta), numel(moving_mean), numel(moving_variance)};
            convertedLayer.LayerName = layerName;
        end
    end
end
