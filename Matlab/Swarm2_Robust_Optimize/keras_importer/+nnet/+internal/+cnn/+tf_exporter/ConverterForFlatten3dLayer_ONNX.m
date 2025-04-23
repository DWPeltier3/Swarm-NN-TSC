classdef ConverterForFlatten3dLayer_ONNX < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The layer supports these input formats:
    % SSSCB --> 111CB   (keepdims=true)

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"SSSC", "SSSCB"}
                    % BSSSC --> B111C in TF. The handwritten custom layer
                    % does the equivalent of a row-major flattening on a
                    % BCHWD tensor, because ONNX uses channels_first. In
                    % the exported TF model we're using channels_last, or
                    % BHWDC. So in our exported code, we need to first
                    % permute to channels_first before doing the native TF
                    % row-major Flattening.
                    convertedLayer.layerCode = [
                        sprintf("bchwd = layers.Permute((4,1,2,3))(%s)", this.InputTensorName); % permute to channels_first
                        sprintf("%s = layers.Reshape((1,1,1,-1))(bchwd)", this.OutputTensorName);
                        ];
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end