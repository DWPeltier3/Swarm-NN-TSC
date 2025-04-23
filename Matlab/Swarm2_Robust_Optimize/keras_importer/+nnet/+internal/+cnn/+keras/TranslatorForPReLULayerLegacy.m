classdef TranslatorForPReLULayerLegacy < nnet.internal.cnn.keras.LayerTranslator
        
    % Copyright 2023 The MathWorks, Inc.
    
    properties(Constant)
        WeightNames = {'alpha'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % Only name and weights will be translated from the config. 
            % The raw weights are saved to a layer parameter called RawAlpha
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
                    alpha = mean(RawAlpha, 'all'); 
                    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedPReLUParameterSize', name);
                else 
                    alpha = RawAlpha; 
                end
            else
                RawAlpha = []; 
                alpha = 0;
            end
            
            newLayer = nnet.keras.layer.PreluLayer(name, alpha); 
            newLayer.RawAlpha = RawAlpha; 
            NNTLayers = {newLayer}; 
        end
    end
end

