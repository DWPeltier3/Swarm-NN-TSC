classdef ConverterForSelfAttentionLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % The keras MultiHeadAttention layer operates on all spatial
            % and time dimensions in the input by default, that is, when
            % attention_axes=None. In contrast, DLT selfAttention layer
            % operates on spatial or time dimension only and it errors if
            % the input has both dimensions. It also errors if the input
            % has more than one spatial dimension.

            % Support table of cases:
            %
            % Data (input --> output):
            %     DLT formats          TF formats
            %  ---------------      ---------------
            % 1   CBT --> CBT         BTC --> BTC
            % 2   SCB --> SCB         BSC --> BSC
            %
            % Padding mask input:
            % 1   CBT                 BTT

            % Get TF hyperparameters:
            inSize = this.InputSize{1};
            outSize = this.OutputSize{1};
            numInputFeatures = inSize(end);
            numOutputFeatures = outSize(end);
            numHeads = this.Layer.NumHeads;
            numKeyChannels = this.Layer.NumKeyChannels;
            dropoutProb = this.Layer.DropoutProbability;

            numValueChannels = this.Layer.NumValueChannels;
            if ~isnumeric(numValueChannels)
                % numValueChannels is 'auto'
                numValueChannels = numKeyChannels;
            end
            numKeyChannelsPerHead = numKeyChannels/numHeads;
            numValueChannelsPerHead = numValueChannels/numHeads;

            % Generate code:
            layerName = this.OutputTensorName(1) + "_";
            convertedLayer.LayerName = layerName;

            inputNames = sprintf("%s, %s", this.InputTensorName(1), this.InputTensorName(1));
            if this.Layer.HasPaddingMaskInput
                maskInputName = this.InputTensorName(2) + "_processed";
                maskCode = sprintf("%s = AttentionMaskLayer()(%s)", maskInputName, this.InputTensorName(2)); % BTC --> BTT
                argListIn = sprintf("%s, attention_mask=%s", inputNames, maskInputName);
                convertedLayer.customLayerNames = "AttentionMaskLayer";
            else
                maskCode = string.empty;
                argListIn = inputNames;
            end

            if strcmp(this.Layer.AttentionMask, "causal")
                %  argListIn = sprintf("%s, use_causal_mask=True", argListIn);

                % use_causal_mask call argument is only supported in the
                % latest version (2.10) of tensorflow and we are not able
                % to test it until we upgrade the tensorflow version used
                % in our test infrastructure.
                convertedLayer.Success = false;
                return
            end

            if this.Layer.HasScoresOutput
                argListOut = sprintf("%s, %s", this.OutputTensorName(1), this.OutputTensorName(2));
                argListIn = sprintf("%s, return_attention_scores=True", argListIn);
            else
                argListOut = sprintf("%s", this.OutputTensorName(1));
            end

            selfAttentionCode = sprintf("%s = layers.MultiHeadAttention(%d, %d, value_dim=%d, output_shape=%d, dropout=%f, " + ...
                "name=""%s"")(%s)", argListOut, numHeads, numKeyChannelsPerHead, numValueChannelsPerHead, numOutputFeatures, ...
                dropoutProb, layerName, argListIn);

            convertedLayer.layerCode = [maskCode; selfAttentionCode];

            % Create TF weights:
            keyWeightsSplitSize = [numKeyChannelsPerHead, numHeads, numInputFeatures];
            keyBiasSplitSize = [numKeyChannelsPerHead, numHeads];
            valueWeightsSplitSize = [numValueChannelsPerHead, numHeads, numInputFeatures];
            valueBiasSplitSize = [numValueChannelsPerHead, numHeads];
            outputWeightsSplitSize = [numOutputFeatures, numValueChannelsPerHead, numHeads];

            kerasQueryWeights = reshape(this.Layer.QueryWeights, keyWeightsSplitSize);
            kerasKeyWeights = reshape(this.Layer.KeyWeights, keyWeightsSplitSize);
            kerasValueWeights = reshape(this.Layer.ValueWeights, valueWeightsSplitSize);
            kerasOutputWeights = reshape(this.Layer.OutputWeights, outputWeightsSplitSize);
            kerasQueryBias = reshape(this.Layer.QueryBias, keyBiasSplitSize);
            kerasKeyBias = reshape(this.Layer.KeyBias, keyBiasSplitSize);
            kerasValueBias = reshape(this.Layer.ValueBias, valueBiasSplitSize);
            kerasOutputBias = this.Layer.OutputBias;

            convertedLayer.weightNames = ["query_kernel", "query_bias", "key_kernel", "key_bias", ...
                "value_kernel", "value_bias", "output_kernel", "output_bias"];
            convertedLayer.weightArrays = {kerasQueryWeights; kerasQueryBias; kerasKeyWeights; kerasKeyBias; ...
                kerasValueWeights; kerasValueBias; kerasOutputWeights; kerasOutputBias};

            kerasKeyWeightsShape = flip(keyWeightsSplitSize);
            kerasKeyBiasShape = flip(keyBiasSplitSize);
            convertedLayer.weightShapes = {kerasKeyWeightsShape; kerasKeyBiasShape; kerasKeyWeightsShape; kerasKeyBiasShape; ...
                flip(valueWeightsSplitSize); flip(valueBiasSplitSize); flip(outputWeightsSplitSize); numel(kerasOutputBias)};
        end
    end
end
