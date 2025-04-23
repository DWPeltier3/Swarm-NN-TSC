classdef TranslatorForTFOpLambdaLayer < nnet.internal.cnn.keras.LayerTranslator
        
    % Copyright 2023 The MathWorks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, ~, ~)
            % 
            % LSpec.KerasConfig
            % ans = 
            % 
            % struct with fields:
            % 
            %            name: 'tf.nn.relu_1'
            %       trainable: 1
            %           dtype: 'float32'
            %        function: 'nn.relu'
            %   inbound_nodes: {{4×1 cell}}
            Name = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            lambdaFunction = kerasField(LSpec,'function');
            switch lambdaFunction            
                case 'nn.relu'
                    NNTLayers = { reluLayer('Name', Name) };
                case 'nn.elu'
                    NNTLayers = { eluLayer('Name', Name) };
                case 'nn.softmax'
                    NNTLayers = { softmaxLayer('Name', Name) };
                case 'compat.v1.nn.softmax'
                    NNTLayers = { softmaxLayer('Name', Name) };
                case 'nn.silu'
                    NNTLayers = { swishLayer('Name', Name) };
                case 'nn.gelu'
                    inboundNodes = kerasField(LSpec,'inbound_nodes');
                    if numel(inboundNodes{1}{1}) == 4 && isstruct(inboundNodes{1}{1}{4}) && isfield(inboundNodes{1}{1}{4},'approximate') && inboundNodes{1}{1}{4}.approximate
                        NNTLayers = { geluLayer('Name', Name, 'Approximation', 'tanh')};
                    else
                        NNTLayers = { geluLayer('Name', Name) };
                    end
                case 'nn.softsign'
                    NNTLayers = { nnet.keras.layer.SoftsignLayer(Name) };
                case 'nn.selu'
                    if nnet.internal.cnn.keras.isInstalledRLT
                        NNTLayers = { eluLayer(1.67326324, 'Name', [Name '_elu']), ...
                                      scalingLayer('Name', [Name '_scale'], 'Scale', 1.05070098)};
                    else
                        Layer = this.createPlaceholderLayer(LSpec, TranslateWeights, Name);
                        NNTLayers = { Layer };
                    end
                case 'math.tanh'
                    NNTLayers = { tanhLayer('Name', Name) };
                case 'math.sigmoid'
                    NNTLayers = { sigmoidLayer('Name', Name) };
                case 'math.softplus'
                    if nnet.internal.cnn.keras.isInstalledRLT
                        NNTLayers = { softplusLayer('Name', Name) };
                    else
                        Layer = this.createPlaceholderLayer(LSpec, TranslateWeights, Name);
                        NNTLayers = { Layer };
                    end
                case 'math.exp'
                    NNTLayers = { functionLayer(@exp, 'Name', Name) };
                otherwise
                    Layer = this.createPlaceholderLayer(LSpec, TranslateWeights, Name);
                    NNTLayers = { Layer };
            end
        end

        function placeholderLayer = createPlaceholderLayer(this, LSpec, TranslateWeights, Name)
            numInputs   = max(1, numel(LSpec.InConns));
            numOutputs  = LSpec.NumOutputs;
            placeholderLayer = nnet.keras.layer.PlaceholderLayer(Name, this.KerasLayerType, ...
                                LSpec.KerasConfig, numInputs, numOutputs);
            if TranslateWeights
                placeholderLayer.Weights = LSpec.Weights;
            end
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if hasKerasField(LSpec, 'function') && isSupportedActivationFunc(this, {LSpec.KerasConfig.function})
                tf = true;
                Message = '';
            elseif hasKerasField(LSpec, 'function')
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedTFOpLambdaLayerFun', LSpec.KerasConfig.function);
            else
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedTFOpLambdaLayerFun', 'No function found.');
            end
        end
        
        function tf = isSupportedActivationFunc(~, FuncStr)
            supportedFunctions = {'nn.relu', 'math.tanh', 'nn.elu', 'math.sigmoid', 'compat.v1.nn.softmax', 'nn.softmax', 'nn.silu', 'nn.gelu', 'math.softplus', 'math.exp', 'nn.softsign', 'nn.selu'};
            tf = ismember(FuncStr, supportedFunctions);
        end
    end
end