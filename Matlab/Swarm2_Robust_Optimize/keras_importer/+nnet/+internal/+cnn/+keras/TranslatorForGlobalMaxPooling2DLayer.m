classdef TranslatorForGlobalMaxPooling2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2021-2022 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % This layer performs downsampling by computing maximum across the 
            % height and width dimension of the input.
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %       trainable: 1
            %     data_format: 'channels_last'
            %            name: 'global_Max_pooling2d_1'
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            % For older versions of TensorFlow, the keepdims parameter does
            % not exist and the behavior is as if keepdims is set to
            % false. Therefore when there is no keepdims field, a flatten 
            % layer is added for producing same results as in TF.
            if hasKerasField(LSpec,'keepdims') && kerasField(LSpec,'keepdims')==1
                NNTLayers = { globalMaxPooling2dLayer('Name', Name) };
            else
                NNTLayers = { globalMaxPooling2dLayer('Name', Name),...
                    flattenLayer('Name', [Name '_flatten']) };
            end
        end
    end
end