classdef TranslatorForSoftmaxLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2018 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %          name: 'softmax_1'
            %     trainable: 1
            %          axis: -1
            NNTLayers = { softmaxLayer('Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if ~ismember(kerasField(LSpec, 'axis'), [-1])
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedAxis2', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end