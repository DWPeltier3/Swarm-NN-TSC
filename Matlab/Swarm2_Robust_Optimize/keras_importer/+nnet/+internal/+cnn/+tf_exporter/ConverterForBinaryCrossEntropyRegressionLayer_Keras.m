classdef ConverterForBinaryCrossEntropyRegressionLayer_Keras <nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The identity function
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName, this.InputTensorName);
        end
    end
end
