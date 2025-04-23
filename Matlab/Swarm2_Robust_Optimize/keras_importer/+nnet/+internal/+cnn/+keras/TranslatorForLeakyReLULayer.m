classdef TranslatorForLeakyReLULayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %     trainable: 1
            %          name: 'leaky_re_lu_1'
            %         alpha: 0.2000
            %
            % --OR--
            %
            % ans = 
            %   struct with fields:
            %          name: 'leaky_re_lu_1'
            %     trainable: 1
            %         alpha: [1×1 struct]
            %
            % -- WHERE ALPHA IS --
            %
            %   struct with fields:
            %     value: 0.2000
            %      type: 'ndarray'
            alph = kerasField(LSpec, 'alpha');
            if isstruct(alph)
                alpha = alph.value;
            else
                alpha = alph;
            end
            NNTLayers = { leakyReluLayer(alpha, 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
    end
end