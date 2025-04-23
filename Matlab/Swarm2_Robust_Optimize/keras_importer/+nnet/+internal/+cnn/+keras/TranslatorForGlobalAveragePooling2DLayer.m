classdef TranslatorForGlobalAveragePooling2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % It's full-image average pooling within each channel, turning a [m,n,p]
            % tensor into a [1,p]
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %       trainable: 1
            %     data_format: 'channels_last'
            %            name: 'global_average_pooling2d_1'
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NNTLayers = { globalAveragePooling2dLayer('Name', Name) };
        end
    end
end