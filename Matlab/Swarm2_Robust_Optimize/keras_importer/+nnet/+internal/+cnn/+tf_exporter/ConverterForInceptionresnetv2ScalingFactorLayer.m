classdef ConverterForInceptionresnetv2ScalingFactorLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            scale = this.Layer.Scale;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.Lambda(lambda x: x * %f)(%s)", this.OutputTensorName, scale, this.InputTensorName);   % A lambda layer.
        end
    end
end
