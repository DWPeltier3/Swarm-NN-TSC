classdef ConverterForBiLSTMLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B]T for the main input and CB for state inputs. 

    methods
        function convertedLayer = toTensorflow(this)
            try
                layerName = this.OutputTensorName(1) + "_";
                convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
                convertedLayer.LayerName = layerName;
                convertedLayer.layerCode = genbiLSTMCode(this, layerName);
                convertedLayer = setbiLSTMWeights(this, convertedLayer);
                if this.Layer.HasStateInputs
                    convertedLayer.customLayerNames = "SplitLayer";
                end
            catch
                convertedLayer.Success = false;
                return
            end
        end

        function code = genbiLSTMCode(this, layerName)
            % Supported input formats are S*CB[T].
            bilstmInputName = this.InputTensorName(1) + "_bilstm_input";
            initialAssignmentCode = sprintf("%s = %s", bilstmInputName, this.InputTensorName(1));
            % If any S are present, they are all flattened MATLAB-style.
            % This is accomplished by a Permute and Flatten in TF. If there
            % is a T dimension, these are time-distributed.
            if contains(this.InputFormat(1), 'S')
                dataNumDims = numel(this.InputSize{1});
                flattenCode = [
                    kerasCodeLine(this, bilstmInputName, bilstmInputName, ...
                    "layers.Permute", "(%s)", {join(string(dataNumDims:-1:1), ',')}, ...
                    "", contains(this.InputFormat(1), 'T'))
                    kerasCodeLine(this, bilstmInputName, bilstmInputName, ...
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
                reshapeCode1 = sprintf("%s = layers.Reshape((1,-1))(%s)", bilstmInputName, bilstmInputName);                        % BC --> B1C
                reshapeCode2 = sprintf("%s = layers.Reshape((-1,))(%s)", this.OutputTensorName(1), this.OutputTensorName(1));   % B1C --> BC
            else
                reshapeCode1 = string.empty;
                reshapeCode2 = string.empty;
            end
            % When HasStateInputs, the DLT layer receives 3 inputs: X,H,C,
            % but the TF layer requires 5: Y, Hfwd, Cfwd, Hbwd, Cbwd. In
            % that case we generate code to split the two DLT-like input
            % tensors H and C into the 4 required ones.
            if this.Layer.HasStateInputs
                splitStateCode = [
                    sprintf("Hfwd, Hbwd = SplitLayer(2, axis=1)(%s)\t\t# Split hidden state input", this.InputTensorName(2))
                    sprintf("Cfwd, Cbwd = SplitLayer(2, axis=1)(%s)\t\t# Split cell state input", this.InputTensorName(3))
                    ];
            else
                splitStateCode = string.empty;
            end
            % Code to call biLSTM layer
            [argListOut, constructorArgs, argListIn] = bilstmArgStrings(this, bilstmInputName);
            bilstmCode = sprintf("%s = layers.Bidirectional(layers.LSTM(%s), name=""%s"")(%s)", argListOut, constructorArgs, layerName, argListIn);
            % When HasStateOutputs, the TF code outputs a list of 5
            % tensors: Y, Hfwd, Cfwd, Hbwd, Cbwd, whereas the DLT net
            % outputs 3: Y,H,C. In that case we generate code to create
            % the DLT-like concatenated output tensors.
            if this.Layer.HasStateOutputs
                concatStateCode = [
                    sprintf("%s = layers.Concatenate(axis=1)([Hfwd, Hbwd])\t\t# Concatenate hidden state outputs", this.OutputTensorName(2))
                    sprintf("%s = layers.Concatenate(axis=1)([Cfwd, Cbwd])\t\t# Concatenate cell state outputs", this.OutputTensorName(3))
                    ];
            else
                concatStateCode = string.empty;
            end
            % Assemble the code
            code = [
                initialAssignmentCode
                flattenCode
                reshapeCode1
                splitStateCode
                bilstmCode
                concatStateCode
                reshapeCode2
                ];
        end

        function [argListOut, constructorArgs, argListIn] = bilstmArgStrings(this, bilstmInputName)
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
            if this.Layer.HasStateInputs
                argListIn = sprintf("%s, initial_state=[Hfwd, Cfwd, Hbwd, Cbwd]", bilstmInputName);
            else
                argListIn = sprintf("%s", bilstmInputName);
            end
            if this.Layer.HasStateOutputs
                argListOut = sprintf("%s, Hfwd, Cfwd, Hbwd, Cbwd", this.OutputTensorName(1));
            else
                argListOut = sprintf("%s", this.OutputTensorName(1));
            end
            constructorArgs = sprintf("%d, activation='%s', recurrent_activation='%s', return_sequences=%s, return_state=%s",...
                units, activation, recurrent_activation, return_sequences, return_state);
        end

        function convertedLayer = setbiLSTMWeights(this, convertedLayer)
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
            [forwardInd, backwardInd] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH, 'lstm');
            ifcoForward = forwardInd([iInd fInd cInd oInd]);
            ifcoBackward = backwardInd([iInd fInd cInd oInd]);

            convertedLayer.weightNames = ["FWDkernel", "FWDrecurrent_kernel", "FWDbias", "BWDkernel", "BWDrecurrent_kernel", "BWDbias"];

            % First save the forward weights, then the backward:

            % kernel: DLT is size [4*H,C] col-maj. TF is the transpose
            % row-maj, so no permute necessary.
            FWDkernel = this.Layer.InputWeights(ifcoForward,:);

            % recurrent_kernel: DLT is size [4*H,C] col-maj. TF is the
            % transpose row-maj, so no permute necessary.
            FWDrecurrent_kernel = this.Layer.RecurrentWeights(ifcoForward,:);

            % DLT bias is a vector of length 4*nH. TF bias is size
            % (4*H,)
            FWDbias = this.Layer.Bias(ifcoForward);

            % kernel: DLT is size [4*H,C] col-maj. TF is the transpose
            % row-maj, so no permute necessary.
            BWDkernel = this.Layer.InputWeights(ifcoBackward,:);
            kernelShape = flip(size(BWDkernel, 1:2));

            % recurrent_kernel: DLT is size [4*H,C] col-maj. TF is the
            % transpose row-maj, so no permute necessary.
            BWDrecurrent_kernel = this.Layer.RecurrentWeights(ifcoBackward,:);
            recurrent_kernelShape = flip(size(BWDrecurrent_kernel, 1:2));

            % DLT bias is a vector of length 4*nH. TF bias is size
            % (4*H,)
            BWDbias = this.Layer.Bias(ifcoBackward);
            biasShape = 4*nH;

            convertedLayer.weightArrays  = {FWDkernel; FWDrecurrent_kernel; FWDbias; BWDkernel; BWDrecurrent_kernel; BWDbias};
            convertedLayer.weightShapes  = {kernelShape; recurrent_kernelShape; biasShape; kernelShape; recurrent_kernelShape; biasShape};
        end
    end
end