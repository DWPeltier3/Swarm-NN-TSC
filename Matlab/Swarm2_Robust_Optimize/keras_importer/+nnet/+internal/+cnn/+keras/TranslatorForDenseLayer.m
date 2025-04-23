classdef TranslatorForDenseLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %                     name: 'dense_1'
            %     activity_regularizer: []
            %       kernel_initializer: [1×1 struct]
            %                 use_bias: 1
            %               activation: 'linear'
            %                    units: 20
            %          bias_constraint: []
            %         bias_regularizer: []
            %       kernel_regularizer: []
            %                trainable: 1
            %         bias_initializer: [1×1 struct]
            %        kernel_constraint: []
            % Parse a LayerConfig struct for a 'Dense' layer and output 1 or 2 layers
            % implementing the Dense layer as closely as possible.
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumUnits    = kerasField(LSpec, 'units');
            UseBias     = logical(kerasField(LSpec, 'use_bias'));
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            % Create layer
            FC = fullyConnectedLayer(NumUnits, ...
                'Name', LayerName,...
                'BiasLearnRateFactor', single(UseBias));
            if TranslateWeights
                if UseBias
                    verifyWeights(LSpec, 'bias');
                    FC.Bias = single(LSpec.Weights.bias);
                else
                    FC.Bias = zeros(NumUnits, 1, 'single');
                end
                verifyWeights(LSpec, 'kernel');
                FC.Weights = single(LSpec.Weights.kernel);
            end
            NNTLayers = {FC};
            NNTLayers = maybeAppendActivationLayer(LSpec, NNTLayers);
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            % Check unsupported options:
            if ~isempty(kerasField(LSpec, 'activity_regularizer'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
            
            if LSpec.isTensorFlowLayer
            % Do not generate a layer if the activation is not supported for TF Layers.
                supportedActivations = nnet.internal.cnn.keras.getSupportedActivations(); 
                if hasKerasField(LSpec, 'activation') && ~ismember(kerasField(LSpec, 'activation'), supportedActivations)
                    tf = false; 
                    Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedActivation', ...
                        kerasField(LSpec, 'activation'));
                end
            end
        end
    end
end

