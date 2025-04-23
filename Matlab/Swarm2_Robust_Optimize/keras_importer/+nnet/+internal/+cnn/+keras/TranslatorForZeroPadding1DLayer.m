classdef TranslatorForZeroPadding1DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2021 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerConfig =
            %   struct with fields:
            %            name: 'zero_padding1d_1'
            %       trainable: 1
            %     data_format: 'channels_last'
            %         padding: 2         
            %
            % padding: int, or tuple of 2 ints
            Amounts = kerasField(LSpec, 'padding')';
            NNTLayers = { nnet.keras.layer.ZeroPadding1dLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name), Amounts(:)) };
        end
    end
end