classdef ConverterForFlattenLayer_ONNX < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % The layer supports these input formats:
    % SSCB --> 11CB             (keepdims=true)

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"SSC", "SSCB"}
                    % BSSC --> B11C in TF. The handwritten custom layer does
                    % the equivalent of a row-major flattening on a BCHW
                    % tensor, because ONNX uses channels_first. In the
                    % exported TF model we're using channels_last, or BHWC.
                    % So in our exported code, we need to first permute to
                    % channels_first before doing the native TF row-major
                    % Flattening.
                    convertedLayer.layerCode = [
                        sprintf("bchw = layers.Permute((3,1,2))(%s)", this.InputTensorName);        % permute to channels_first
                        sprintf("%s = layers.Reshape((1,1,-1))(bchw)", this.OutputTensorName);
                        ];
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end