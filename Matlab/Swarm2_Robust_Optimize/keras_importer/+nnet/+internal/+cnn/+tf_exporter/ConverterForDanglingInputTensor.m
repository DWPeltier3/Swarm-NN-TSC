classdef ConverterForDanglingInputTensor < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this, inputNum)
            % Create a convertedLayer for inputNum of this layer
            tensorSize = this.InputSize{inputNum};
            tensorName = this.InputTensorName(inputNum);
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % Rename the input tensor
            newTensorName = sprintf("%s_input", tensorName);
            convertedLayer.RenameNetworkInputTensor = [tensorName, newTensorName];
            % If the tensor has a T dimension, then we need to include that
            % in the Keras input layer declaration as 'None'
            if contains(this.InputFormat(inputNum), "T")
                maybeNone = "None,";
            else
                maybeNone = "";
            end
            convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(%s%s))", ...
                newTensorName, maybeNone, join(string(tensorSize), ','));
        end
    end
end