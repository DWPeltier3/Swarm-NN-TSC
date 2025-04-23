classdef TranslatorForResizingLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2022 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % This layer resize the input image to the specified height and
            % width.
            % LSpec.KerasConfig
            % ans =
            % struct with fields:  
            % 
            %                  name: 'resizing'
            %                trainable: 1
            %        batch_input_shape: [4×1 double]
            %                    dtype: 'float32'
            %                   height: 28
            %                    width: 24
            %            interpolation: 'bilinear'
            %     crop_to_aspect_ratio: 0
            LayerName = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            OutputSize = [LSpec.KerasConfig.height, LSpec.KerasConfig.width];

            % Other interpolation methods from the keras Resizing layer 
            % are not supported currently. Will default to 'bilinear' for 
            % those methods.
            if hasKerasField(LSpec, 'interpolation') && ismember(kerasField(LSpec,'interpolation'), ["bilinear", "nearest"])
                Method = kerasField(LSpec, 'interpolation');
                if Method=="nearest"
                    iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:ImporterResizeNearest', LSpec.Name);
                end
                NNTLayers = { resize2dLayer('OutputSize', OutputSize, 'Method', Method, 'Name', LayerName) };
            else
                NNTLayers = { resize2dLayer('OutputSize', OutputSize, 'Method', 'bilinear', 'Name', LayerName) };
                iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedInterpolationMethod', LSpec.Name);
            end
        end
        
        function [tf, Message] = canSupportSettings(~, LSpec)
            % resize2dLayer only supports 'nearest' and 'bilinear' method of 
            % interpolation. Other interpolation methods from the keras 
            % Resizing layer are not supported currently, will default to 
            % 'bilinear'. The default method in keras is 'bilinear'. Also, 
            % the crop_to_aspect_ratio feature is not supported by this layer.
            if LSpec.KerasConfig.crop_to_aspect_ratio
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedResizeWithCrop', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end