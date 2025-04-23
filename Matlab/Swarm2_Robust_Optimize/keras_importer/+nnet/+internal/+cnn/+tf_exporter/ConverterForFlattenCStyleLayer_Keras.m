classdef ConverterForFlattenCStyleLayer_Keras < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The layer supports these input formats:
    % SC or SCB -->1CB          (keepdims=true)
    % SSCB --> 11CB             (keepdims=true)
    % SSSC or SSSCB --> 111CB   (keepdims=true)
    % CBT --> CB                (keepdims=false) (The layer first permutes([1 3 2]) into [C T B])

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"SC", "SCB"}
                    % BSC --> B1C in TF
                    convertedLayer.layerCode = sprintf("%s = layers.Reshape((1,-1))(%s)", this.OutputTensorName, this.InputTensorName);
                case {"SSC", "SSCB"}
                    % BSSC --> B11C in TF
                    convertedLayer.layerCode = sprintf("%s = layers.Reshape((1,1,-1))(%s)", this.OutputTensorName, this.InputTensorName);
                case {"SSSC", "SSSCB"}
                    % BSSSC --> B111C in TF
                    convertedLayer.layerCode = sprintf("%s = layers.Reshape((1,1,1,-1))(%s)", this.OutputTensorName, this.InputTensorName);
                case {"CT", "CBT"}
                    % BTC --> BC in TF
                    convertedLayer.layerCode = sprintf("%s = layers.Flatten()(%s)", this.OutputTensorName, this.InputTensorName);
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end