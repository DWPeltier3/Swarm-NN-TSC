classdef TranslatorForGlobalAveragePooling3DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2019 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %       trainable: 1
            %     data_format: 'channels_last'
            %            name: 'global_average_pooling3d_1'
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NNTLayers = { globalAveragePooling3dLayer('Name', Name) };
        end
    end
end