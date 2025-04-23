classdef TranslatorForReshapeLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2020 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            
            NNTLayers = { nnet.keras.layer.FlattenCStyleLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if isfield(LSpec.KerasConfig, 'data_format') && isequal(LSpec.KerasConfig.data_format, 'channels_first')
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoChannelsFirst', LSpec.Name);
            elseif isfield(LSpec.KerasConfig, 'target_shape') && iSpecifyVector(kerasField(LSpec, 'target_shape'))
                tf = true;
                Message = '';
            else 
                tf = false; 
                targetShapeValsGTOne = sum(kerasField(LSpec, 'target_shape') ~= 1, 'all');
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedReshapeLayer', LSpec.Name, targetShapeValsGTOne);
            end
        end
    end
end

function tf = iSpecifyVector(kerasSizes)
    % Count how many non-singleton dimensions of the reshape. If only one,
    % Treat the layer as a flatten.
    tf = sum(kerasSizes ~= 1, 'all') == 1; 
end 
