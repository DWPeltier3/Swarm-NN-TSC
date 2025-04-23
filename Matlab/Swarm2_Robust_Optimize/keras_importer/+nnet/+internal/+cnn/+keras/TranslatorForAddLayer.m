classdef TranslatorForAddLayer < nnet.internal.cnn.keras.LayerTranslator

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
            %          name: 'add_1'
            %     trainable: 1
            NumInputs = numel(LSpec.InConns);
            NNTLayers = { additionLayer(NumInputs, 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
    end
end