classdef ConverterForGlobalAveragePooling2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported MATLAB input formats and behavior:
    %     SSCB --> SSCB = 11CB
    %     SSCBT --> SSCBT = 11CBT
    %     SSCT --> SSCT = 11CT
    %     SCBT --> SCB = 1CB      (spatio-temporal)
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if this.InputFormat=="SCBT"
                % In TF the desired mapping is BTSC --> BSC = B1C. To do
                % this, we do GAP2d to obtain BC, then call
                % layers.Reshape((1,-1)), which means C should be reshaped
                % to 1C.
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.GlobalAveragePooling2D", "", {}, "", false)

                    kerasCodeLine(this, this.OutputTensorName, this.OutputTensorName,...
                    "layers.Reshape", "(1,-1)", {}, "", false)
                    ];
            else
                % Just do a normal GAP, possibly TimeDistributed
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.GlobalAveragePooling2D", "keepdims=True", {}, "", this.layerAnalyzer.IsTemporal);
            end
        end
    end
end