classdef ConverterForZeroPadding2dLayer_Keras < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The layer supports these input formats:
    % SSCB --> SSCB
    % SSC --> SSC
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"SSC", "SSCB"}
                    % Always BCSS in TF.
                    convertedLayer.layerCode = [
                        sprintf("%s = layers.ZeroPadding2D(padding=((%d, %d), (%d, %d)))(%s)", ...
                        this.OutputTensorName, ...
                        this.Layer.Top, this.Layer.Bottom, this.Layer.Left, this.Layer.Right,...
                        this.InputTensorName);
                        ];
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end