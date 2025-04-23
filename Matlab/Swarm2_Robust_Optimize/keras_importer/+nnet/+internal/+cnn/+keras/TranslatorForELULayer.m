classdef TranslatorForELULayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2019 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans = 
            %   struct with fields:
            % 
            %          name: 'elu'
            %     trainable: 1
            %         dtype: 'float32'
            %         alpha: 0.5000
            alpha = kerasField(LSpec, 'alpha');
            NNTLayers = { eluLayer(alpha, 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
    end
end