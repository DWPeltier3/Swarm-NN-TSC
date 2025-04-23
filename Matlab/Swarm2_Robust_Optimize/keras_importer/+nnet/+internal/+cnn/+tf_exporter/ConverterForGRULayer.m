classdef ConverterForGRULayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022-2023 The MathWorks, Inc.

    % Supported input formats: S*CB[T] for the main input and CB for state inputs. 
    % Compared to LSTMLayer, GRU layer has a ResetGateMode argument, it can
    % have 1 or 2 outputs, whereas LSTM can have 1 or 3 outputs. it can
    % have 1 or 2 inputs, whereas LSTM can have 1 or 3 inputs.
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = this.convertGRUWithWeightsToTensorflow(this.Layer.InputWeights, this.Layer.RecurrentWeights);
        end

        function convertedLayer = convertGRUWithWeightsToTensorflow(this, inputWeightsToExport, recurrentWeightsToExport)
            % Base method that is used both by ConverterForGRULayer and
            % ConverterForGRUProjectedLayer 
            
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            try
                layerName = this.OutputTensorName(1) + "_";
                convertedLayer.LayerName = layerName;
                convertedLayer.layerCode = genGRUCode(this, layerName);
                convertedLayer = setGRUWeights(this, convertedLayer, inputWeightsToExport, recurrentWeightsToExport);
            catch
                convertedLayer.Success = false;
                return
            end
        end

        function code = genGRUCode(this, layerName)
            % Supported input formats are S*CB[T].
            gruInputName = this.InputTensorName(1) + "_gru_input";
            initialAssignmentCode = sprintf("%s = %s", gruInputName, this.InputTensorName(1));
            % If any S are present, they are all flattened MATLAB-style.
            % This is accomplished by a Permute and Flatten in TF. If there
            % is a T dimension, these are time-distributed.
            if contains(this.InputFormat(1), 'S')
                dataNumDims = numel(this.InputSize{1});                     % This does not include the B or T dimensions.
                flattenCode = [
                    kerasCodeLine(this, gruInputName, gruInputName, ...
                    "layers.Permute", "(%s)", {join(string(dataNumDims:-1:1), ',')}, ...
                    "", contains(this.InputFormat(1), 'T'))
                    kerasCodeLine(this, gruInputName, gruInputName, ...
                    "layers.Flatten", "", {}, ...
                    "", contains(this.InputFormat(1), 'T'))
                    ];
            else
                flattenCode = string.empty;
            end
            % If the T dimension is missing, then the TF GRU call is
            % wrapped in Reshapes to pass BTC=B1C to the GRU, and then
            % return it to BC afterward.
            if ~contains(this.InputFormat(1), 'T')
                reshapeCode1 = sprintf("%s = layers.Reshape((1,-1))(%s)", gruInputName, gruInputName);                        % BC --> B1C
                reshapeCode2 = sprintf("%s = layers.Reshape((-1,))(%s)", this.OutputTensorName(1), this.OutputTensorName(1));   % B1C --> BC
            else
                reshapeCode1 = string.empty;
                reshapeCode2 = string.empty;
            end
            % Build the GRU call
            [argListOut, constructorArgs, argListIn] = gruArgStrings(this, gruInputName, layerName);
            gruCode = sprintf("%s = layers.GRU(%s)(%s)", argListOut, constructorArgs, argListIn);
            % Assemble the code
            code = [
                initialAssignmentCode
                flattenCode
                reshapeCode1
                gruCode
                reshapeCode2
                ];
        end


        function [argListOut, constructorArgs, argListIn] = gruArgStrings(this, gruInputName, layerName)
            units = this.Layer.NumHiddenUnits;
            switch this.Layer.StateActivationFunction
                case 'tanh'
                    activation = "tanh";
                case 'softsign'
                    activation = "softsign";
                otherwise
                    assert(false);
            end
            switch this.Layer.GateActivationFunction
                case 'sigmoid'
                    recurrent_activation = "sigmoid";
                case 'hard-sigmoid'
                    recurrent_activation = "hard_sigmoid";
                otherwise
                    assert(false);
            end
            switch this.Layer.OutputMode
                case 'sequence'
                    return_sequences = "True";
                case 'last'
                    return_sequences = "False";
                otherwise
                    assert(false);
            end
            if this.Layer.HasStateOutputs
                return_state = "True";
            else
                return_state = "False";
            end
            switch this.Layer.ResetGateMode
                case {'after-multiplication', 'recurrent-bias-after-multiplication'}
                    reset_after = "True";
                case 'before-multiplication'
                    reset_after = "False";
                otherwise
                    assert(false);
            end
            if this.Layer.HasStateInputs
                argListIn = sprintf("%s, initial_state=[%s]", gruInputName, this.InputTensorName(2));
            else
                argListIn = sprintf("%s", gruInputName);
            end
            if this.Layer.HasStateOutputs
                argListOut = sprintf("%s, %s", this.OutputTensorName(1), this.OutputTensorName(2));
            else
                argListOut = sprintf("%s", this.OutputTensorName(1));
            end
            constructorArgs = sprintf("%d, name='%s', activation='%s', recurrent_activation='%s', return_sequences=%s, return_state=%s, reset_after=%s",...
                units, layerName, activation, recurrent_activation, return_sequences, return_state, reset_after);
        end

        function convertedLayer = setGRUWeights(this, convertedLayer, inputWeightsToExport, recurrentWeightsToExport)
            % Get weight indices
            nH = this.Layer.NumHiddenUnits;
            [rInd, zInd, hInd] = nnet.internal.cnn.util.gruGateIndices(nH);
            [forwardInd, ~] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH, 'gru');
            zrcForward = forwardInd([zInd, rInd, hInd]);

            convertedLayer.weightNames = ["kernel", "recurrent_kernel", "bias"];

            % kernel: DLT is size [3*H,C] col-maj. TF is the transpose
            % row-maj, so no permute necessary.
            kernel = inputWeightsToExport(zrcForward,:);
            kernelShape = flip(size(kernel, 1:2));

            % recurrent_kernel: DLT is size [3*H,H] col-maj. TF is the
            % transpose row-maj, so no permute necessary.
            recurrent_kernel = recurrentWeightsToExport(zrcForward,:);
            recurrent_kernelShape = flip(size(recurrent_kernel, 1:2));

            % bias has 3 cases:
            switch this.Layer.ResetGateMode
                case 'after-multiplication'
                    % DLT bias is a vector of length 3*nH. TF bias is size
                    % [2, 3*H] row-maj. We fill just the first row.
                    bias = zeros(3*nH, 2);
                    bias(:,1) = this.Layer.Bias(zrcForward);
                    biasShape = [2, 3*nH];
                case 'recurrent-bias-after-multiplication'
                    % DLT bias is a vector of length 6*nH. TF bias is size
                    % [2, 3*H] row-maj. We fill the rows in order.
                    bias = zeros(3*nH, 2);
                    bias(:,1) = this.Layer.Bias(zrcForward);
                    bias(:,2) = this.Layer.Bias(3*nH + zrcForward);
                    biasShape = [2, 3*nH];
                case 'before-multiplication'
                    % DLT bias is a vector of length 3*nH. TF bias is size
                    % (3*H,)
                    bias = this.Layer.Bias(zrcForward);
                    biasShape = 3*nH;
                otherwise
                    assert(false)
            end
            % Tell TF that the sizes are transposed.
            convertedLayer.weightArrays  = {kernel; recurrent_kernel; bias};
            convertedLayer.weightShapes  = {kernelShape; recurrent_kernelShape; biasShape};
        end
    end
end