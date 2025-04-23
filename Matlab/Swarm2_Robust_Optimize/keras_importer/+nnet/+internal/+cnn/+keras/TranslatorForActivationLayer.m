 classdef TranslatorForActivationLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2022 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %      trainable: 1
            %           name: 'activation_1'
            %     activation: 'relu'
            LayerName = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            addSuffix = ~isequal(LSpec.Type, 'Activation');
            [isLayer, layerName] = getActivationNameFromLSpec(LSpec);
            
            % If activation is specified as a Keras activation layer  
            % create it using the respective activation layer translator
            if isLayer
                % First create a new LayerSpec for the activation layer
                activationLayerConfig = LSpec.KerasConfig.('activation').config;
                activationLayerSpec = nnet.internal.cnn.keras.LayerSpec();
                activationLayerSpec.KerasConfig = activationLayerConfig;
                if isfield(activationLayerConfig,'name')
                    activationLayerSpec.Name = [LayerName '_' activationLayerConfig.name];
                else
                    activationLayerSpec.Name = [LayerName '_' lower(layerName)];
                end               
                
                % Now call the translator to get the activation layer 
                activationLayerTranslator = nnet.internal.cnn.keras.LayerTranslator.create(layerName, activationLayerSpec);
                NNTLayers = translate(activationLayerTranslator, activationLayerSpec, false, false);    
            else
                % Else the activation is specified as a string
                % hence directly create an activation layer
                switch lower(layerName)
                    case 'linear'
                        NNTLayers = {};
                    case 'relu'
                        if addSuffix
                            LayerName = [LayerName '_relu'];
                        end
                        NNTLayers = { reluLayer('Name', LayerName) };
                    case 'relu6'
                        if addSuffix
                            LayerName = [LayerName '_relu6'];
                        end
                        NNTLayers = { clippedReluLayer(6, 'Name', LayerName) };
                    case 'sigmoid'
                        if addSuffix
                            LayerName = [LayerName '_sigmoid'];
                        end
                        NNTLayers = { sigmoidLayer('Name', LayerName) };
                    case 'softmax'
                        if addSuffix
                            LayerName = [LayerName '_softmax'];
                        end
                        NNTLayers = { softmaxLayer('Name', LayerName) };
                    case 'tanh'
                        if addSuffix
                            LayerName = [LayerName '_tanh'];
                        end
                        NNTLayers = { tanhLayer('Name', LayerName) };
                    case 'elu'
                        if addSuffix
                            LayerName = [LayerName '_elu'];
                        end
                        NNTLayers = { eluLayer(1, 'Name', LayerName) };
                    case 'swish'
                        if addSuffix
                            LayerName = [LayerName '_swish']; 
                        end 
                        NNTLayers = { swishLayer('Name', LayerName) };
                    case 'gelu'
                        if addSuffix
                            LayerName = [LayerName '_gelu'];
                        end
                        if hasKerasField(LSpec, 'approximate')
                            useApproximation = logical(kerasField(LSpec, 'approximate'));
                            if isempty(useApproximation) || ~useApproximation
                                NNTLayers = { geluLayer('Name', LayerName) };
                            else
                                NNTLayers = { geluLayer('Approximation', 'tanh', 'Name', LayerName) };
                            end
                        else
                            NNTLayers = { geluLayer('Name', LayerName) };
                        end
                    otherwise
                        assert(false);
                end
            end
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            [isLayer, layerName] = getActivationNameFromLSpec(LSpec);
            % Check if activation is either a layer or a supported
            % activation function string. If it is a layer we will check if
            % it is a supported layer later on in the Layer Translator
            % hence we don't, explicitly check for the layer being
            % supported over here.
            if isLayer || ismember(lower(layerName), {'linear', 'relu', 'relu6', 'sigmoid', 'softmax', 'tanh', 'elu', 'swish', 'gelu'})
                    tf = true;
                    Message = '';
            else
                    tf = false;
                    Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedActivation', ...
                        kerasField(LSpec, 'activation'));
            end
        end
    end
end