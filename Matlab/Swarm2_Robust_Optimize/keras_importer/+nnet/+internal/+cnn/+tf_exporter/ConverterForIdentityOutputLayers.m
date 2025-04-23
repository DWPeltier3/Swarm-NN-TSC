classdef ConverterForIdentityOutputLayers < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % This is used for all output layers that simply pass through their
    % input tensor unchanged. Generate code to set the output tensor equal
    % to the input tensor
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName, this.InputTensorName);
        end
    end
end
