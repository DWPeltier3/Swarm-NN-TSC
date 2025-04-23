classdef ConverterForDepthConcatenationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)
            % Concatenate over the Channel dimension. According to our
            % tensor correspondence table (see base class), in all our TF
            % tensors, Channel is present, and is the last dimension.
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            tfDim = -1;
            convertedLayer.layerCode = sprintf("%s = layers.Concatenate(axis=%d)([%s])", ...
                this.OutputTensorName, tfDim, join(this.InputTensorName, ', '));
        end
    end
end