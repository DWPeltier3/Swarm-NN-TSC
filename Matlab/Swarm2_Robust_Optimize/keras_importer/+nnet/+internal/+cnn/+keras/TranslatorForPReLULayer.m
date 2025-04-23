classdef TranslatorForPReLULayer < nnet.internal.cnn.keras.LayerTranslator
        
    % Copyright 2019-2023 The MathWorks, Inc.
    
    properties(Constant)
        WeightNames = {'alpha'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % Only name and weights will be translated from the config. 
            % LSpec.KerasConfig
            % ans = 
            % 
            %   struct with fields:
            % 
            %                  name: 'p_re_lu_1'
            %      alpha_constraint: []
            %             trainable: 1
            %     alpha_regularizer: []
            %     alpha_initializer: [1×1 struct]
            %           shared_axes: [] 
            name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            if TranslateWeights
                RawAlpha = LSpec.Weights.alpha;
                if ~isscalar(RawAlpha)
                    % Can't support vector alpha as its shape depends upon input
                    % rank. This information is not available during
                    % translation
                    alpha = mean(RawAlpha, 'all'); 
                    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedPReLUParameterSize', name);
                else 
                    alpha = RawAlpha; 
                end
            else
                alpha = 0;
            end

            newLayer = preluLayer('Name', name, 'Alpha', alpha);
            NNTLayers = {newLayer}; 
        end
    end
end
