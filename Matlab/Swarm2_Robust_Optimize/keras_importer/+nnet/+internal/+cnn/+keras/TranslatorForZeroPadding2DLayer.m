classdef TranslatorForZeroPadding2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerConfig =
            %   struct with fields:
            %            name: 'zero_padding2d_1'
            %       trainable: 1
            %     data_format: 'channels_last'
            %         padding: [2×2 double]            
            %
            % padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            % * If int: the same symmetric padding is applied to width and height.
            % * If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: 
            %   (symmetric_height_pad, symmetric_width_pad).
            % * If tuple of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))
            % In MATLAB, the matrix case would be 
            % [top_pad, bottom_pad; 
            %  left_pad, right_pad]
            Amounts = kerasField(LSpec, 'padding')';
            NNTLayers = { nnet.keras.layer.ZeroPadding2dLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name), Amounts(:)) };
        end
    end
end