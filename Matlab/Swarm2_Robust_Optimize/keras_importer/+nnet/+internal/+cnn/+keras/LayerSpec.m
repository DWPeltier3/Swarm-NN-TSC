classdef LayerSpec < handle

% Copyright 2017-2023 The MathWorks, Inc.

    properties
        Name
        Type
        KerasConfig
        Translator              % A LayerTranslator.
        InConns = {};           % Only used for DAG layers.
        NumOutputs = 1;          % Only used for DAG layers. 
        Weights = [];  % A struct when there are weights.
        IsInput3D = false;      % A flag keeping track of whether the input tensor is 5D or not
                                % For 5D tensor, data is N x H x W x D x C (data is 3D)
                                % For 4D tensor, data is N x H x W x C (data is 2D)
        IsFeatureInput = false;
        IsTimeDistributed = false;
        IsSequenceInput = false;
        Has1DLayers = false;
        isTensorFlowLayer = false;
        TimeDistributedName = '';
        SubmodelName = '';
    end
    
    methods(Static)
        function this = fromBaseLayer(KerasLayer,SubmodelName, isTFLayer)
            % KerasLayer is a KerasLayerInsideSequentialModel.
            this = nnet.internal.cnn.keras.LayerSpec();
            this.Name = KerasLayer.Config.KerasStruct.name;
            this.Type = KerasLayer.ClassName;
            this.KerasConfig = KerasLayer.Config.KerasStruct;
            this.IsTimeDistributed = KerasLayer.IsTimeDistributed;
            this.SubmodelName = SubmodelName;
            this.IsInput3D = false;
            this.isTensorFlowLayer = isTFLayer;
        end
        
        function this = fromTimeDistributedBaseLayer(KerasLayer,SubmodelName, TimeDistributedName, isTFLayer)
            % KerasLayer is a KerasLayerInsideSequentialModel.
            this = nnet.internal.cnn.keras.LayerSpec();
            this.Name = KerasLayer.Config.KerasStruct.name;
            this.Type = KerasLayer.ClassName;
            this.KerasConfig = KerasLayer.Config.KerasStruct;
            this.IsTimeDistributed = true;
            this.TimeDistributedName = TimeDistributedName;
            this.SubmodelName = SubmodelName;
            this.IsInput3D = false;
            this.isTensorFlowLayer = isTFLayer;
        end        
    end
    
    methods
        function NNTLayers = translateLayer(this, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % Translate a single Keras layer into a list of NNT layers, which may
            % include an activation layer.
            NNTLayers = translate(this.Translator, this, TranslateWeights, TranslateTrainingParams, UserImageInputSize);
        end
        
        function NNTLayers = translateDAGLayer(this, OutputTensors, TrainingConfig, TranslateWeights, TranslateTrainingParams, UserImageInputSize, isRNN)
            % Translate layer (which may add an activation layer), then maybe add an
            % output layer.
            %checkUnsupportedDAGLayers(this);

            NNTLayers = translateLayer(this, TranslateWeights, TranslateTrainingParams, UserImageInputSize);
            NNTLayers = appendOutputLayerIfNeeded(this, NNTLayers, OutputTensors, TrainingConfig, isRNN);
        end
        
        function NNTLayers = maybeAppendActivationLayer(this, NNTLayers)
            % When a layer has an 'activation' option, this function can be called to
            % append a layer for it.
            if hasKerasField(this, 'activation')
                ActivationTranslator = nnet.internal.cnn.keras.LayerTranslator.create('Activation', this, this.isTensorFlowLayer);
                LayerCell = translate(ActivationTranslator, this, false, false);
                if ~isempty(LayerCell) && isa(LayerCell{1}, 'nnet.keras.layer.PlaceholderLayer') 
                    LayerCell{1}.Name = [LayerCell{1}.Name '_' kerasField(this, 'activation')]; 
                end
                NNTLayers = [NNTLayers, LayerCell];
            end
        end
        
        function verifyWeights(this, FieldName)
            % Error if there are no weights or if the weights lack the FieldName
            if ~isstruct(this.Weights)
                error(message('nnet_cnn_kerasimporter:keras_importer:NoWeightsInLayer', this.Name));
            elseif ~isfield(this.Weights, FieldName)
                error(message('nnet_cnn_kerasimporter:keras_importer:NoWeightField', FieldName, this.Name));
            end
        end

        function tf = hasKerasField(this, FieldName)
            tf = isfield(this.KerasConfig, FieldName);
        end
        
        function Value = kerasField(this, FieldName)
            if hasKerasField(this, FieldName)
                Value = this.KerasConfig.(FieldName);
                if isstruct(Value) && isfield(Value, 'class_name') && strcmp(Value.class_name, '__tuple__')
                    Value = Value.items; 
                end 
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MissingKerasField', this.Name, this.Type, FieldName)));
            end
        end
        
        function [isLayer, activationName] = getActivationNameFromLSpec(this)
            if hasKerasField(this, 'activation')
                % This could be a struct if it is a Keras Layer or a string
                maybeStruct = this.KerasConfig.('activation');
                if isstruct(maybeStruct)
                % Struct should have field 'class_name' containing the
                % activation name
                isLayer = true;
                    if isfield(maybeStruct, 'class_name')
                        activationName = maybeStruct.class_name;
                    else
                        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MissingKerasField', this.Name, this.Type, 'class_name')));
                    end
                else
                % String which is the activation name
                    isLayer = false;
                    activationName = maybeStruct;
                end
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MissingKerasField', this.Name, this.Type, 'activation')));
            end
        end
        
        function this = setInputLayerFlagInLayerSpec(this, isInput3D, isFeatureInput, isRNN, has1DLayers)
            this.IsInput3D       = isInput3D;
            this.IsFeatureInput  = isFeatureInput;
            if isRNN
                this.IsSequenceInput = true;
            end
            this.Has1DLayers     = has1DLayers;
        end
        
        function this = setTimeDistributedFlagInLayerSpec(this, isTimeDistributed)
            this.IsTimeDistributed = isTimeDistributed;            
        end
        
        % Set Translator based on KerasLayerType
        function this =  setTranslatorInLayerSpec(this, ImportWeights, WeightFile, H5Info, isDAGNetwork)
            this.Translator = nnet.internal.cnn.keras.LayerTranslator.create(this.Type, this, isDAGNetwork);
            if ImportWeights
                this.Weights = importWeights(this.Translator, this.Name, this.SubmodelName, WeightFile, H5Info, this.TimeDistributedName);
            end
        end
    end
    
    methods(Access=protected)
        function checkUnsupportedDAGLayers(this)
            if isequal(this.Type, 'LSTM')
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NoLSTMInDAG')));
            end
        end
        
        function NNTLayers = appendOutputLayerIfNeeded(this, NNTLayersFromKerasLayer, OutputTensors, TrainingConfig, isRNN)
            % If LayerSpec is mentioned in the network's OutputTensors, then add an
            % output layer.
            NNTLayers = NNTLayersFromKerasLayer;
            if ~isempty(OutputTensors) && ~isempty(TrainingConfig)
                OutputLayerNames = cellfun(@(T)T.FromLayerName, OutputTensors, 'UniformOutput', false);
                % If input to an output layer is a custom layer, prepend
                % the name with 'tf_op_layer_'
                if ~isempty(NNTLayers) && isa(NNTLayers{1}, 'nnet.keras.layer.PlaceholderLayer') && ismember(this.Type,{'TensorFlowOpLayer'})
                    [~,pos] = ismember(['tf_op_layer_' this.Name], OutputLayerNames);
                else
                    [~,pos] = ismember(this.Name, OutputLayerNames);
                end
                if pos > 0
                    outputLayer = DAGOutputLayer(OutputTensors{pos}, TrainingConfig, isRNN);
                    if ~isempty(outputLayer)
                        NNTLayers{end+1} = outputLayer;
                    end
                end
            end
        end
    end
end

function NNTLayer = DAGOutputLayer(OutputTensor, TrainingConfig, isRNN)
CurLayerName = OutputTensor.FromLayerName;
if isstruct(TrainingConfig.loss)
    Loss = iConvertLossStructToString(TrainingConfig, CurLayerName); 
else
    Loss = TrainingConfig.loss;
end

NNTName = nnet.internal.cnn.keras.makeNNTName(CurLayerName);
if isempty(Loss)
    % The model has a TrainingConfig but the loss information is empty.
    % This can happen if someone calls the 'compile()' method of a keras 
    % model, without specifying any arguments (i.e., loss function), before saving it.
    NNTLayer = [];
    return;
end
switch Loss
    case {'categorical_crossentropy', 'sparse_categorical_crossentropy'}
        NNTLayer = classificationLayer('Name', sprintf('ClassificationLayer_%s', NNTName));
    case {'binary_crossentropy'}
        NNTLayer = nnet.keras.layer.BinaryCrossEntropyRegressionLayer(sprintf('BinaryCrossEntropyRegressionLayer_%s', NNTName), isRNN);
    case {'mean_squared_error', 'mse'}
        NNTLayer = regressionLayer('Name', sprintf('RegressionLayer_%s', NNTName));    
    otherwise
        if ~strcmp(Loss, 'None')
            % By convention, 'None' means a multiple output network with no
            % loss at all. If it is something else, it is just an
            % unsupported loss. 
            iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedLoss', Loss);
        end 
        NNTLayer = nnet.keras.layer.PlaceholderOutputLayer([NNTName '_OutputLayer_PLACEHOLDER']);
end
end

function loss = iConvertLossStructToString(TrainingConfig, layerName) 
    % If the training config is a struct in MATLAB, there are two possible 
    % causes. Either there is more than one loss in an MO network or the 
    % loss is a class. This function converts both cases into an 
    % appropriate string. 
    
    if isfield(TrainingConfig.loss, layerName) 
        % If the network is MO, look-up the current layer loss.  
        loss = TrainingConfig.loss.(layerName);
    else
        loss = TrainingConfig.loss; 
    end
    
    % If the current layer loss is a class, convert it to the equivalent string. 
    if isfield(loss, 'class_name')
        switch loss.class_name
            case {'CategoricalCrossentropy', 'SparseCategoricalCrossentropy'}
                loss = 'categorical_crossentropy'; 
            case {'BinaryCrossentropy'}
                loss = 'binary_crossentropy'; 
            case {'MeanSquaredError'}
                loss = 'mean_squared_error';
            otherwise 
                % Unrecognized class name will produce an unsupported output
                % layer. 
                loss = 'None'; 
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end
