classdef TranslatorForGlobalAveragePooling1DLayer < nnet.internal.cnn.keras.LayerTranslator
    % Copyright 2021 The Mathworks, Inc.

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
            %            name: 'global_average_pooling1d_1'
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NNTLayers = { globalAveragePooling1dLayer('Name', Name) };
        end
    end
end

