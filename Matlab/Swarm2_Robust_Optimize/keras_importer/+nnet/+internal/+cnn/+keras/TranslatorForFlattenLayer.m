classdef TranslatorForFlattenLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2017 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(~, LSpec, ~, ~, ~)
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %     trainable: 1
            %          name: 'flatten_1'
            
            % KERAS 2.1.6:
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %     trainable: 1
            %          name: 'flatten_1'
            %          data_format: 'channels_first'
            if LSpec.isTensorFlowLayer
                NNTLayers = { nnet.keras.layer.FlattenCStyleTFLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name))};
            else
                NNTLayers = { nnet.keras.layer.FlattenCStyleLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
            end
        end
        
        function [tf, Message] = canSupportSettings(~, LSpec)
            if isfield(LSpec.KerasConfig, 'data_format') && isequal(LSpec.KerasConfig.data_format, 'channels_first')
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoChannelsFirst', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end