classdef ConverterForCustomRegressionLayer <nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % This is for CUSTOM regression output layers only.
    % Supports 2D and 3D image input tensors
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName, this.InputTensorName);
        end
    end
end
