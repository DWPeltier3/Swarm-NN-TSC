classdef ConverterForFlatten3dInto2dLayer_ONNX < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The layer supports these input formats:
    % SSSCB --> CB   (keepdims=false)

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"SSSC", "SSSCB"}
                    % BSSSC --> BC in TF. The handwritten custom layer does
                    % the equivalent of a row-major flattening on a BCHWD
                    % tensor, because ONNX uses channels_first. In the
                    % exported TF model we're using channels_last, or
                    % BHWDC. So in our exported code, we need to first
                    % permute to channels_first before doing the native TF
                    % row-major Flattening.
                    convertedLayer.layerCode = [
                        sprintf("bchwd = layers.Permute((4,1,2,3))(%s)", this.InputTensorName); % permute to channels_first
                        sprintf("%s = layers.Flatten()(bchwd)", this.OutputTensorName);         
                        ];
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end