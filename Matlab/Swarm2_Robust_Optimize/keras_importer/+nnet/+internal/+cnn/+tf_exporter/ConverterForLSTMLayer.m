classdef ConverterForLSTMLayer < nnet.internal.cnn.tf_exporter.ConverterForLSTMLayer_Base

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = toTensorflow@nnet.internal.cnn.tf_exporter.ConverterForLSTMLayer_Base(this, ...
                this.Layer.InputWeights, this.Layer.RecurrentWeights);
        end
    end
end