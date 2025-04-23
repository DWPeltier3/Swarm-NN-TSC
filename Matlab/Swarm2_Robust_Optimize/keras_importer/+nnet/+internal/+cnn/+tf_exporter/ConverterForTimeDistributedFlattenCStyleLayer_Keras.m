classdef ConverterForTimeDistributedFlattenCStyleLayer_Keras < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The layer supports these input formats:
    % SSC[B] --> CB             (keepdims=false)
    % other --> identity

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"SSC", "SSCB"}
                    % BSSC --> BC in TF. A simple row-major Flattening.
                    convertedLayer.layerCode = [
                        sprintf("%s = layers.Flatten()(%s)", this.OutputTensorName, this.InputTensorName);
                        ];
                otherwise
                    % Identity
                    convertedLayer.layerCode = [
                        sprintf("%s = %s", this.OutputTensorName, this.InputTensorName);
                        ];
            end
        end
    end
end