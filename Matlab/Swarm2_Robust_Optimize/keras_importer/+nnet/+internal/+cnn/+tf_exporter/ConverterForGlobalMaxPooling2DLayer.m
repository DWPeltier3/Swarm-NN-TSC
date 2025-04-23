classdef ConverterForGlobalMaxPooling2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B][T], SC[B]T 
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "layers.GlobalMaxPool2D", "keepdims=True", {}, "", this.layerAnalyzer.IsTemporal);
        end
    end
end