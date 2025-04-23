classdef TranslatorForInputLayers < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2017-2021 The MathWorks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            %   struct with fields:
            %                  name: 'input_1'
            %                 dtype: 'float32'
            %     batch_input_shape: [4x1 double] or [5x1 double]
            %                sparse: 0
            if hasKerasField(LSpec, 'batch_input_shape')
                KerasInputShape = kerasField(LSpec, 'batch_input_shape');
            elseif ~isempty(UserImageInputSize)
                KerasInputShape = [NaN UserImageInputSize(:)'];
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ImageInputSizeNeeded')));
            end
            
            % Check unsupported options
            if kerasField(LSpec, 'sparse') ~= 0
                iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:NoSparseInput', LSpec.Name);
            end

            NNTLayers  = { nnet.keras.layer.PlaceholderInputLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name), KerasInputShape, LSpec.KerasConfig) };
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end
