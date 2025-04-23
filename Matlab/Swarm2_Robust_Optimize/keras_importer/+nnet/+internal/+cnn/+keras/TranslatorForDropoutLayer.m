classdef TranslatorForDropoutLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %          name: 'dropout_1'
            %     trainable: 1
            %          rate: 0.4320
            NNTLayers = { dropoutLayer(kerasField(LSpec, 'rate'), 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
    end
end