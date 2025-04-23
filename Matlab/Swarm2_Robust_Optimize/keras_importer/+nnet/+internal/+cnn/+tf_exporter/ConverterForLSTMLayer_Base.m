classdef ConverterForLSTMLayer_Base < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*CB[T] for the main input and CB for state
    % inputs. Compared to GRULayer, LSTM layer has no ResetGateMode
    % argument, it can have 1 or 3 outputs, whereas GRU can have 1 or 2
    % outputs. It can have 1 or 3 inputs, whereas GRU can have 1 or 2
    % inputs.
    methods
        % toTensorflow takes extra arguments, the plain (not projected)
        % LSTM weights to be exported.
        function convertedLayer = toTensorflow(this, inputWeightsToExport, recurrentWeightsToExport)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            try
                layerName = this.OutputTensorName(1) + "_";
                convertedLayer.LayerName = layerName;
                convertedLayer.layerCode = genLSTMCode(this, layerName);
                convertedLayer = setLSTMWeights(this, convertedLayer, inputWeightsToExport, recurrentWeightsToExport);
            catch
                convertedLayer.Success = false;
                return
            end
        end

        function code = genLSTMCode(this, layerName)
            % Supported input formats are S*CB[T].
            lstmInputName = this.InputTensorName(1) + "_lstm_input";
            initialAssignmentCode = sprintf("%s = %s", lstmInputName, this.InputTensorName(1));
            % If any S are present, they are all flattened MATLAB-style.
            % This is accomplished by a Permute and Flatten in TF. If there
            % is a T dimension, these are time-distributed.
            if contains(this.InputFormat(1), 'S')
                dataNumDims = numel(this.InputSize{1});
                flattenCode = [
                    kerasCodeLine(this, lstmInputName, lstmInputName, ...
                    "layers.Permute", "(%s)", {join(string(dataNumDims:-1:1), ',')}, ...
                    "", contains(this.InputFormat(1), 'T'))
                    kerasCodeLine(this, lstmInputName, lstmInputName, ...
                    "layers.Flatten", "", {}, ...
                    "", contains(this.InputFormat(1), 'T'))
                    ];
            else
                flattenCode = string.empty;
            end
            % If the T dimension is missing, then the TF LSTM call is
            % wrapped in Reshapes to pass BTC=B1C to the LSTM, and then
            % return it to BC afterward.
            if ~contains(this.InputFormat(1), 'T')
                reshapeCode1 = sprintf("%s = layers.Reshape((1,-1))(%s)", lstmInputName, lstmInputName);                        % BC --> B1C
                reshapeCode2 = sprintf("%s = layers.Reshape((-1,))(%s)", this.OutputTensorName(1), this.OutputTensorName(1));   % B1C --> BC
            else
                reshapeCode1 = string.empty;
                reshapeCode2 = string.empty;
            end
            % Build the LSTM call
            [argListOut, constructorArgs, argListIn] = lstmArgStrings(this, lstmInputName, layerName);
            lstmCode = sprintf("%s = layers.LSTM(%s)(%s)", argListOut, constructorArgs, argListIn);
            % Assemble the code
            code = [
                initialAssignmentCode
                flattenCode
                reshapeCode1
                lstmCode
                reshapeCode2
                ];
        end

        function [argListOut, constructorArgs, argListIn] = lstmArgStrings(this, lstmInputName, layerName)
            units = this.Layer.NumHiddenUnits;
            switch this.Layer.StateActivationFunction
                case 'tanh'
                    activation = "tanh";
                case 'relu'
                    activation = "relu";
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
            if this.Layer.HasStateInputs
                argListIn = sprintf("%s, initial_state=[%s, %s]", lstmInputName, this.InputTensorName(2), this.InputTensorName(3));
            else
                argListIn = sprintf("%s", lstmInputName);
            end
            if this.Layer.HasStateOutputs
                argListOut = sprintf("%s, %s, %s", this.OutputTensorName(1), this.OutputTensorName(2), this.OutputTensorName(3));
            else
                argListOut = sprintf("%s", this.OutputTensorName(1));
            end
            constructorArgs = sprintf("%d, name='%s', activation='%s', recurrent_activation='%s', return_sequences=%s, return_state=%s",...
                units, layerName, activation, recurrent_activation, return_sequences, return_state);
        end

        function convertedLayer = setLSTMWeights(this, convertedLayer, inputWeightsToExport, recurrentWeightsToExport)
            % This method uses the passed weight arguments as the weights
            % to be exported, so it can support both lstmLayer and
            % lstmProjectedLayer.
            %
            % Source code from TF/Keras that shows weight ordering (from
            % file keras/layers/recurrent_v2.py):
            %
            %             def step(cell_inputs, cell_states):
            %             """Step function that will be used by Keras RNN backend."""
            %             h_tm1 = cell_states[0]  # previous memory state
            %             c_tm1 = cell_states[1]  # previous carry state
            %
            %             z = backend.dot(cell_inputs, kernel)
            %             z += backend.dot(h_tm1, recurrent_kernel)
            %             z = backend.bias_add(z, bias)
            %
            %             z0, z1, z2, z3 = tf.split(z, 4, axis=1)
            %
            %             i = tf.sigmoid(z0)
            %             f = tf.sigmoid(z1)
            %             c = f * c_tm1 + i * tf.tanh(z2)
            %             o = tf.sigmoid(z3)
            %
            %             h = o * tf.tanh(c)
            %             return h, [h, c]

            % Get weight indices. As we see from the above code, the
            % Keras ordering needs to be IFCO, so we arrange that below:
            nH = this.Layer.NumHiddenUnits;
            [cInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(nH);
            [forwardInd, ~] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH, 'lstm');
            ifcoForward = forwardInd([iInd fInd cInd oInd]);

            convertedLayer.weightNames = ["kernel", "recurrent_kernel", "bias"];

            % kernel: DLT is size [4*H,C] col-maj. TF is the transpose
            % row-maj, so no permute necessary.
            kernel = inputWeightsToExport(ifcoForward,:);
            kernelShape = flip(size(kernel, 1:2));

            % recurrent_kernel: DLT is size [4*H,H] col-maj. TF is the
            % transpose row-maj, so no permute necessary.
            recurrent_kernel = recurrentWeightsToExport(ifcoForward,:);
            recurrent_kernelShape = flip(size(recurrent_kernel, 1:2));

            % DLT bias is a vector of length 4*nH. TF bias is size
            % (4*H,)
            bias = this.Layer.Bias(ifcoForward);
            biasShape = 4*nH;

            convertedLayer.weightArrays  = {kernel; recurrent_kernel; bias};
            convertedLayer.weightShapes  = {kernelShape; recurrent_kernelShape; biasShape};
        end
    end
end