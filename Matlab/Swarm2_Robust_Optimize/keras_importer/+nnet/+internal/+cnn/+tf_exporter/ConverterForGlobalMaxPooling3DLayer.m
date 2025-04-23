classdef ConverterForGlobalMaxPooling3DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSSC[B][T], SSC[B]T 
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "layers.GlobalMaxPool3D", "keepdims=True", {}, "", this.layerAnalyzer.IsTemporal);
        end
    end
end