classdef ConverterForMultiplicationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all

    % MATLAB doc: "The size of the inputs to the multiplication layer must
    % be either same across all dimensions or same across at least one
    % dimension with other dimensions as singleton dimensions."
    % (broadcasting)

    % tf.keras doc: "It takes as input a list of tensors, all of the same
    % shape, and returns a single tensor (also of the same shape)."
    % (no broadcasting)

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            [bcode, multInputNames] = broadcastingCode(this);
            convertedLayer.layerCode = [
                bcode
                sprintf("%s = layers.Multiply()([%s])", this.OutputTensorName, join(multInputNames, ', '))
                ];
            if ~isempty(bcode)
                convertedLayer.customLayerNames = "BroadcastLayer";
            end
        end

        function [code, tensorNames] = broadcastingCode(this)
            % Add broadcasting layers for any input tensors that are not
            % the same size as the output tensor
            code = string.empty;
            tensorNames = this.InputTensorName;
            numBTDims = 1 + contains(this.OutputFormat, "T");   % 2 if temporal, 1 if not.
            for i=1:numel(this.InputTensorName)
                if ~isequal(this.InputSize{i}, this.OutputSize{1})
                    newName = tensorNames(i) + "_broadcast";
                    tensorNames(i) = newName;
                    observationShape = this.OutputSize{1};     % In DLT, this already excludes B and T.
                    code = [
                        code
                        sprintf("%s = BroadcastLayer(%d,[%s])(%s)", newName, numBTDims, join(string(observationShape), ','), this.InputTensorName(i))
                        ];
                end
            end
        end
    end
end
