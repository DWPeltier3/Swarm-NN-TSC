classdef TranslatorForReLULayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2018 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %     name: 're_lu_1'
            %     trainable: 1
            %     max_value: 0.1000 or [] if user passed 'None' in keras
            %     negative_slope: 0.1000 or [] if user passed 'None' in keras
            maxValue = kerasField(LSpec, 'max_value');
            if hasKerasField(LSpec, 'negative_slope') 
                scale = kerasField(LSpec, 'negative_slope');
                if scale==0
                    scale = [];
                end
            else
                scale = [];
            end
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            % Handle 4 cases:
            if isempty(maxValue) && isempty(scale)
                % Relu
                NNTLayers = { reluLayer('Name', Name) };
            elseif ~isempty(maxValue) && isempty(scale)
                % Clipped relu
                NNTLayers = { clippedReluLayer(maxValue, 'Name', Name) };
            elseif isempty(maxValue) && ~isempty(scale)
                % Leaky relu
                NNTLayers = { leakyReluLayer(scale, 'Name', Name) };
            elseif ~isempty(maxValue) && ~isempty(scale)
                % Leaky relu followed by Clip
                NNTLayers = { leakyReluLayer(scale, 'Name', [Name '_LeakyRelu']),...
                    nnet.keras.layer.ClipLayer([Name '_Clip'], -Inf, maxValue) };
            else
                assert(false);
            end
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if hasKerasField(LSpec, 'threshold') && kerasField(LSpec, 'threshold') ~= 0
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedReluThreshold', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end