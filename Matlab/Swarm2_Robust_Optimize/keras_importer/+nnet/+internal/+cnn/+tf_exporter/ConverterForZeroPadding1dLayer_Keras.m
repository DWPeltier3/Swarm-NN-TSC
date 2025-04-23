classdef ConverterForZeroPadding1dLayer_Keras < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022-2023 The MathWorks, Inc.

    % The layer supports these input formats:
    % CBT --> CBT
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"CBT", "CT"}
                    % BTC --> BTC in TF.
                    convertedLayer.layerCode = [
                        sprintf("%s = layers.ZeroPadding1D(padding=(%d, %d))(%s)", ...
                        this.OutputTensorName, this.Layer.leftPad, this.Layer.rightPad, this.InputTensorName);
                        ];
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end
