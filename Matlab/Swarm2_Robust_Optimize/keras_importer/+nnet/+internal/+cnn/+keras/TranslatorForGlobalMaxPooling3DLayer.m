classdef TranslatorForGlobalMaxPooling3DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2022 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % This layer performs downsampling by computing maximum across the 
            % height, width, and depth dimension of the input.
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %            name: 'global_max_pooling3d'
            %       trainable: 1
            %           dtype: 'float32'
            %     data_format: 'channels_last'
            %        keepdims: 1
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            % For older versions of TensorFlow, the keepdims parameter does
            % not exist and the behavior is as if keepdims is set to
            % false. Therefore when there is no keepdims field, a flatten 
            % layer is added for producing same results as in TF.
            if hasKerasField(LSpec,'keepdims') && kerasField(LSpec,'keepdims')==1
                NNTLayers = { globalMaxPooling3dLayer('Name', Name) };
            else
                NNTLayers = { globalMaxPooling3dLayer('Name', Name),...
                    flattenLayer('Name', [Name '_flatten']) };
            end

        end

    end
end