classdef TranslatorForMultiplyLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2020 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %          name: 'multiply_1'
            %     trainable: 1
            NumInputs = numel(LSpec.InConns);
            NNTLayers = { multiplicationLayer(NumInputs, 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
    end
end