classdef ConverterForImageInputLayer < nnet.internal.cnn.tf_exporter.ConverterForImageInputLayers

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)

            if this.Layer.SplitComplexInputs % SplitComplexInputs is not supported
                error(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedSplitComplexInputs', ...
                    this.Layer.Name));
            end

            convertedLayer = toTensorflow@nnet.internal.cnn.tf_exporter.ConverterForImageInputLayers(this);
        end
    end
end
