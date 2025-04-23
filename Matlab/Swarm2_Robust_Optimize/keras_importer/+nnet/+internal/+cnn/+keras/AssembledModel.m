classdef AssembledModel < handle
    
    % Copyright 2017-2023 The MathWorks, Inc.
    
    properties
        ModelType                   % Either 'Sequential' or 'DAG'.
        LayerSpecs                  % LayerSpec objects.
        TrainingConfig;             % Used to determine output layer type.
        WeightsImported = false;    % Logical
        InputLayerIndices = [];     % Only used for DAGs.
        OutputTensors = {};         % Only used for DAGs.
        isDAG = false;
        isTimeDistributed = false;
        isTensorFlowModel = false;
    end
    
    methods
        function this = AssembledModel(KM, WeightFile)
            
            % KM is a ParsedKerasModel.
            if nargin == 1
                ImportWeights = false;
                WeightFile = '';
                H5Info = [];
            else
                ImportWeights = true;
                H5Info = h5info(WeightFile);    % Precompute here for speed.
            end
            this.TrainingConfig   = KM.TrainingConfig;
            this.WeightsImported  = ImportWeights;
            if isa(KM.Model, 'nnet.internal.cnn.keras.KerasSequentialModel')
                this.ModelType    = 'Sequential';
                this.isDAG = KM.Model.isDAG;
            else
                % KM.Model is a KerasDAGModel
                assert(isa(KM.Model, 'nnet.internal.cnn.keras.KerasDAGModel'))
                this.ModelType = 'DAG';
            end
            this.isTimeDistributed = KM.Model.isTimeDistributed;
            this.isTensorFlowModel = KM.Model.isTensorFlowModel;
            [this.LayerSpecs, this.InputLayerIndices, this.OutputTensors] = flatten(KM.Model);
            if ~this.isTensorFlowModel
                % If the first layer is an inputlayer, mark it as
                % timedistribtued so that we can create a sequence of images as
                % input.
                if isequal(this.LayerSpecs{1}.Type, 'InputLayer')
                    this.LayerSpecs{1}.IsSequenceInput = this.isTimeDistributed;
                end
                
                this.LayerSpecs   = cellfun(@(KLayer)setInputLayerFlagInLayerSpec(KLayer,...
                    isInput3D(this), isFeatureInput(this),isRNN(this),has1DLayer(this)), this.LayerSpecs,'UniformOutput',false);

            end
            
            %Setting Translator to Layer Spec after setting input layer
            %type
            this.LayerSpecs = cellfun(@(KLayer) setTranslatorInLayerSpec(KLayer, ImportWeights, WeightFile, H5Info, ~isempty(this.TrainingConfig)), ...
                this.LayerSpecs,'UniformOutput', false);
            
        end
        
        function Layers = translateAssembledModel(this, TrainingConfig, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            if this.isTimeDistributed
                this.LayerSpecs = reorderInputLayerWithSequenceFoldingLayer(this.LayerSpecs);
            end
            switch this.ModelType
                case 'Sequential'
                    Layers = translateSequentialModel(this, TrainingConfig, TranslateWeights, TranslateTrainingParams, UserImageInputSize);
                case 'DAG'
                    if ~this.isTensorFlowModel
                        checkSupportedInputConfiguration(this);
                    end
                    countOutputsForLSpec(this);
                    Layers = translateDAGModel(this, TranslateWeights, TranslateTrainingParams, UserImageInputSize);
                otherwise
                    assert(false);
            end
        end
        
    end
    
    methods(Access=protected)
        function countOutputsForLSpec(this)
            % This method loops through each LSpec and populates its
            % NumOutputs property. 
            
            % Maps name -> lspec index
            LSpecIdx = containers.Map(); 
            for i = 1:numel(this.LayerSpecs)
                if ~isKey(LSpecIdx, this.LayerSpecs{i}.Name)
                    LSpecIdx(this.LayerSpecs{i}.Name) = i;
                end 
                
                for InConnIdx = 1:numel(this.LayerSpecs{i}.InConns)
                    inputLayerName = this.LayerSpecs{i}.InConns{InConnIdx}.FromLayerName; 
                    re = regexp(inputLayerName, '^tf_op_layer_', 'end'); 
                    if ~isempty(re) 
                        inputLayerName = inputLayerName(re+1:end); 
                    end 
                    inputLSpec = this.LayerSpecs{LSpecIdx(inputLayerName)}; 
                    
                    inputLayerOutputNum = this.LayerSpecs{i}.InConns{InConnIdx}.FromOutputNum; 
                    inputLSpec.NumOutputs = max(inputLSpec.NumOutputs, inputLayerOutputNum); 
                end
                
                % check if this layer is an output layer 
                for curOutputTensor = 1:numel(this.OutputTensors)
                    if strcmp(this.LayerSpecs{i}.Name, this.OutputTensors{curOutputTensor}.FromLayerName)
                        this.LayerSpecs{i}.NumOutputs = max(this.LayerSpecs{i}.NumOutputs, this.OutputTensors{curOutputTensor}.FromOutputNum); 
                    end
                end
            end
        end
        
        function checkSupportedInputConfiguration(this)
            inputDimensionality = zeros(1, numel(this.InputLayerIndices)); 
            numSequentialInputs = 0; 
            for i = 1:numel(this.InputLayerIndices)
                curInput = this.LayerSpecs{this.InputLayerIndices(i)}; 
                if curInput.IsSequenceInput
                    % Count number of sequence input layers.
                    numSequentialInputs = numSequentialInputs + 1; 
                end
                % Track image input dimensionality
                inputDimensionality(i) = numel(kerasField(curInput, 'batch_input_shape')); 
            end

            if numSequentialInputs > 0 && (numel(this.InputLayerIndices) > 1 || numel(this.OutputTensors) > 1)
                % MI or MO networks cannot have sequence input layers. 
                error(message('nnet_cnn_kerasimporter:keras_importer:MIMONetworkWithSequenceInput'));
            end
        end
        
        function tf = isRNN(this)
            tf = any(cellfun(@(LS)ismember(LS.Type, {'LSTM', 'Bidirectional',...
                'CuDNNLSTM', 'GRU', 'CuDNNGRU','Conv1D','MaxPooling1D','AveragePooling1D','GlobalAveragePooling1D','GlobalMaxPooling1D'}), this.LayerSpecs, 'UniformOutput', true));
        end
        
        function tf = has1DLayer(this)
            tf = any(cellfun(@(ls)ismember(ls.Type, {'Conv1D','MaxPooling1D','AveragePooling1D','GlobalAveragePooling1D','GlobalMaxPooling1D'}), this.LayerSpecs));
        end
        function tf = isInput3D(this)
            if ~this.isTimeDistributed
                KerasInputShape = kerasField(this.LayerSpecs{1}, 'batch_input_shape');
                tf = (numel(KerasInputShape) == 5); % input is of the form [n h w d c]
            else
                tf = false;
            end
        end
        
        function tf = isFeatureInput(this)
            if ~this.isTimeDistributed
                KerasInputShape = kerasField(this.LayerSpecs{1},'batch_input_shape');
                tf = (numel(KerasInputShape)==2); % input is of the form [n c]
            else
                tf = false;
            end
        end
 
        %% Sequential model
        function NNTLayerGraph = translateSequentialModel(this, TrainingConfig, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            checkDataFormat(this);
            NNTLayerGroups = cellfun(@(LSpec)translateDAGLayer(LSpec, this.OutputTensors, TrainingConfig, TranslateWeights,...
                TranslateTrainingParams, UserImageInputSize,isRNN(this)),...
                this.LayerSpecs, 'UniformOutput', false);
            NNTLayerGraph = connectSequentialLayerGroups(this, NNTLayerGroups, UserImageInputSize);
            NNTLayerGraph = removeDisconnectedLayers(NNTLayerGraph);
            if ~this.isDAG
                NNTLayerGraph = NNTLayerGraph.Layers;
            end
        end
        
        function checkDataFormat(this)
            % make sure no layer has data_format=channels_first.
            for i = 1:numel(this.LayerSpecs)
                LSpec = this.LayerSpecs{i};
                if hasKerasField(LSpec, 'data_format') && isequal(kerasField(LSpec, 'data_format'), 'channels_first')
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NoChannelsFirst', LSpec.Name)));
                end
            end
        end
        
        function NNTLayer = createSequentialInputLayer(this, UserImageInputSize)
            % Examine the first non-input layer.
            LayerSpec1 = this.LayerSpecs{1};
            if hasKerasField(LayerSpec1, 'batch_input_shape')
                KerasInputShape = kerasField(LayerSpec1, 'batch_input_shape');
            elseif ~isempty(UserImageInputSize)
                KerasInputShape = [NaN UserImageInputSize(:)'];
 			else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ImageInputSizeNeeded')));
            end
            
            if this.isTimeDistributed
                % If the input is a sequence of images and at the same time
                % timedistributed layer is applied.
                assert(numel(KerasInputShape) == 5);
                ShouldBeNaN	= KerasInputShape(1);
                Timesteps   = KerasInputShape(2);
                Height      = KerasInputShape(3);
                Width       = KerasInputShape(4);
                Channel     = KerasInputShape(5);
                NNTLayer    = sequenceInputLayer([Height, Width, Channel], 'Name', 'SequenceInputLayer');
            else
                % If any layer indicates a recurrent network, create a
                % sequenceInputLayer, otherwise create an imageInputLayer
                if strcmp(this.LayerSpecs{1}.Type, 'Embedding') 
                    assert(numel(KerasInputShape) == 2);
                    ShouldBeNaN = KerasInputShape(1);
                    Timesteps   = KerasInputShape(2);
                    Width       = 1;
                    if any(cellfun(@(ls)ismember(ls.Type, {'Conv1D','MaxPooling1D','AveragePooling1D',...
                            'GlobalAveragePooling1D','GlobalMaxPooling1D'}), this.LayerSpecs))  && ~isnan(Timesteps)
                        NNTLayer    = sequenceInputLayer(Width, 'Name', 'SequenceInputLayer','MinLength',Timesteps);
                    else
                        NNTLayer    = sequenceInputLayer(Width, 'Name', 'SequenceInputLayer');
                    end
                elseif any(cellfun(@(ls)ismember(ls.Type, {'LSTM', 'Bidirectional', 'CuDNNLSTM', 'GRU',...
                        'CuDNNGRU','Conv1D','MaxPooling1D','AveragePooling1D','GlobalAveragePooling1D','GlobalMaxPooling1D'}), this.LayerSpecs))
                    % RNN layers could reside in the middle of the network.
                    assert(numel(KerasInputShape) == 3);
                    ShouldBeNaN	= KerasInputShape(1);
                    Timesteps   = KerasInputShape(2);
                    Width       = KerasInputShape(3);
                    if any(cellfun(@(ls)ismember(ls.Type, {'Conv1D','MaxPooling1D','AveragePooling1D','GlobalAveragePooling1D','GlobalMaxPooling1D'}), this.LayerSpecs))  && ~isnan(Timesteps)
                        NNTLayer    = sequenceInputLayer(Width, 'Name', 'SequenceInputLayer','MinLength',Timesteps);
                    else
                        NNTLayer    = sequenceInputLayer(Width, 'Name', 'SequenceInputLayer');
                    end
                else
                    switch numel(KerasInputShape)
                        case 2
                            % Each input pattern is just a vector. Use the [height,width] syntax
                            % for the first argument to imageInputLayer. Make the vector a "row image":
                            NNTInputShape = KerasInputShape(2);                            
                        case 3
                            % 2D Image input without channels specified: [NaN height width], implying 1
                            % channel.
                            NNTInputShape = nnet.internal.cnn.keras.determineImageInputSize(KerasInputShape, UserImageInputSize);
                        case 4
                            % Assume it's an 2D image input layer:
                            % Keras inputShape: (NaN, rows, cols, channels)
                            NNTInputShape = nnet.internal.cnn.keras.determineImageInputSize(KerasInputShape, UserImageInputSize);
                        case 5
                            % Assume it's an 3D image input layer:
                            % Keras inputShape: (NaN, rows, cols, depth, channels)
                            NNTInputShape = nnet.internal.cnn.keras.determineImageInputSize(KerasInputShape, UserImageInputSize);                            
                        otherwise
                            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:BadInputDim', num2str(numel(KerasInputShape)-1))));
                    end
                    if any(isnan(NNTInputShape))
                        if numel(NNTInputShape) > 3
                            NNTLayer   = nnet.keras.layer.PlaceholderInputLayer('Image3DInputLayer', KerasInputShape); 
                        else
                            NNTLayer   = nnet.keras.layer.PlaceholderInputLayer('ImageInputLayer', KerasInputShape);
                        end
                    else
                        if numel(NNTInputShape) > 3
                            % NNTInputShape is of the form [h w d c]
                            NNTLayer = image3dInputLayer(NNTInputShape(:)', 'Name', 'Image3DInputLayer', 'Normalization', 'none');
                        elseif numel(NNTInputShape) > 1
                            % NNTInputShape is of the form [h w c]
                            NNTLayer = imageInputLayer(NNTInputShape(:)', 'Name', 'ImageInputLayer', 'Normalization', 'none');
                        else
                            %NNTInputShape if of the form [c]
                            NNTLayer = featureInputLayer(NNTInputShape, 'Name', 'FeatureInputLayer');
                        end
                    end
                end
            end
        end
        
        function NNTLayerGraph = connectSequentialLayerGroups(this, NNTLayerGroups, UserImageInputSize)
            % removeUnusedInputLayers(): We need to do this before adding input layers
            % to a LayerGraph because there can be at most one input layer in a
            % LayerGraph.
            [NNTLayerGroups, this.LayerSpecs] = removeUnusedInputLayers(NNTLayerGroups, this.LayerSpecs);
            LayerList = [ NNTLayerGroups{:} ];
            NNTLayerGraph = layerGraph();
            if ~isequal(this.LayerSpecs{1}.Type, 'InputLayer')
                InputLayer = createSequentialInputLayer(this, UserImageInputSize);
                NNTLayerGraph = addLayers(NNTLayerGraph, InputLayer);
            end
            for L = 1:numel(LayerList)
                NNTLayerGraph = addLayers(NNTLayerGraph, LayerList{L});
            end
            if ~isequal(this.LayerSpecs{1}.Type, 'InputLayer')
            	NNTLayerGraph = connectLayers(NNTLayerGraph, InputLayer.Name, LayerList{1}.Name);
            end
            NNTLayerGraph = addWithinGroupConnections(NNTLayerGraph, this.LayerSpecs, NNTLayerGroups);
            NNTLayerGraph = addBetweenGroupConnections(NNTLayerGraph, this.LayerSpecs, NNTLayerGroups);
        end

        %% DAG model
        function NNTLayerGraph = translateDAGModel(AM, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            checkDataFormat(AM);
            % if numel(AM.LayerSpecs) > 1 && strcmp(AM.LayerSpecs{1}.Type, 'InputLayer')
            %     AM.LayerSpecs = AM.LayerSpecs(2:end);
            %     AM.LayerSpecs{1}.InConns = {};
            % end
            if isempty(AM.TrainingConfig)
                NNTLayerGroups = cellfun(@(LSpec)translateDAGLayer(LSpec, AM.OutputTensors, AM.TrainingConfig, TranslateWeights,...
                TranslateTrainingParams, UserImageInputSize, false),...
                AM.LayerSpecs, 'UniformOutput', false);
            else
                rnnFlag = isRNN(AM);
                NNTLayerGroups = cellfun(@(LSpec)translateDAGLayer(LSpec, AM.OutputTensors, AM.TrainingConfig, TranslateWeights,...
                TranslateTrainingParams, UserImageInputSize, rnnFlag),...
                AM.LayerSpecs, 'UniformOutput', false);
            end
            NNTLayerGraph = connectDAGLayerGroups(NNTLayerGroups, AM.LayerSpecs);
            NNTLayerGraph = removeDisconnectedLayers(NNTLayerGraph);
        end
    end
end

function LayerSpecs = reorderInputLayerWithSequenceFoldingLayer(LayerSpecs)
% Only for TimeDistributed Wrapped Layer, when a TimeDistributed wrapper
% contains a DAG model, the input layer of this DAG model will be between
% sequenceFoldingLayer and sequenceUnfoldingLayer, in order to translate
% input layer, switch the position of input layer and sequenceFoldingLayer
for L = 1:numel(LayerSpecs)
    if isempty(LayerSpecs{L}.InConns)
        continue;
    end
    SourceIdx = findLayerSpecIdx(LayerSpecs{L}.InConns{1}.FromLayerName, LayerSpecs);
    if isequal(LayerSpecs{L}.Type, 'InputLayer') && isequal(LayerSpecs{SourceIdx}.Type, 'TimeDistributedIn')
        NextIdx = findLayerSpecInConnsIdx(LayerSpecs{L}.Name, LayerSpecs);
        SourceInConns = LayerSpecs{L}.InConns;
        LayerSpecs{L}.InConns = LayerSpecs{SourceIdx}.InConns;
        LayerSpecs{SourceIdx}.InConns = LayerSpecs{NextIdx(1)}.InConns;
        for i = 1:numel(NextIdx)
            LayerSpecs{NextIdx(i)}.InConns = SourceInConns;
        end
        % Switch the position
        [LayerSpecs{L}, LayerSpecs{SourceIdx}] = LayerSpecs{[SourceIdx, L]};
        % Set IsTimeDistributed Flag to be zero and IsSequenceInput Flag to
        % be one
        LayerSpecs{SourceIdx}.IsTimeDistributed = false;
        LayerSpecs{SourceIdx}.IsSequenceInput = true;
    end
end
end

function SourceIdx = findLayerSpecInConnsIdx(LayerName, LayerSpecs)
SourceIdx = find(cellfun(@(Spec)~isempty(Spec.InConns) && isequal(Spec.InConns{1}.FromLayerName, LayerName), LayerSpecs));
end

function NNTLayerGraph = connectDAGLayerGroups(NNTLayerGroups, LayerSpecs)
LayerList = [ NNTLayerGroups{:} ];
NNTLayerGraph = layerGraph();
for L = 1:numel(LayerList)
    NNTLayerGraph = addLayers(NNTLayerGraph, LayerList{L});
end
NNTLayerGraph = addWithinGroupConnections(NNTLayerGraph, LayerSpecs, NNTLayerGroups);
NNTLayerGraph = addBetweenGroupConnections(NNTLayerGraph, LayerSpecs, NNTLayerGroups);
end


function tf = isDestination(destination, layerName)
    if isequal(destination, layerName)
        tf = true;
    elseif startsWith(destination,layerName)
        tf = (destination(strlength(layerName) + 1) == '/');
    else 
        tf = false;
    end
end


function [NNTLayerGroups, LayerSpecs] = removeUnusedInputLayers(NNTLayerGroups, LayerSpecs)
AllSendingLayers = iAllSendingLayerNames(LayerSpecs);
UnusedInputLayerIndices = cellfun(@(LS)isequal(LS.Type, 'InputLayer') && ~ismember(LS.Name, AllSendingLayers),...
    LayerSpecs);
NNTLayerGroups(UnusedInputLayerIndices) = [];
LayerSpecs(UnusedInputLayerIndices) = [];
end

function Names = iAllSendingLayerNames(LSpecs)
% Returns a cell array of the names of all layers that ever appear as an
% InConn of another layer.
Names = {};
for i = 1:numel(LSpecs)
    SendersToLSpec = cellfun(@(Tens)Tens.FromLayerName, LSpecs{i}.InConns, 'UniformOutput', false);
    Names = [Names, SendersToLSpec(:)'];
end
end

function NNTLayerGraph = addWithinGroupConnections(NNTLayerGraph, LayerSpecs, NNTLayerGroups)
for group = 1:numel(NNTLayerGroups)
    for toLayer = 2:numel(NNTLayerGroups{group})
        FromNNTName = NNTLayerGroups{group}{toLayer-1}.Name;
        ToNNTName = NNTLayerGroups{group}{toLayer}.Name;
        NNTLayerGraph = connectLayers(NNTLayerGraph, FromNNTName, ToNNTName);
    end
end
end

function NNTLayerGraph = addBetweenGroupConnections(NNTLayerGraph, LayerSpecs, NNTLayerGroups)
for L = 1:numel(LayerSpecs)
    if ~isempty(LayerSpecs{L}.InConns) && ~isempty(NNTLayerGroups{L}) && ~isequal(LayerSpecs{L}.Type, 'InputLayer')
        if numel(LayerSpecs{L}.InConns) == 1
            % Connect the source of this InConn to layer L.
            [NNTSourceName, NNTSourceNum, SourceIdx] = findInConnSource(LayerSpecs{L}.InConns{1}, LayerSpecs, NNTLayerGroups);
            % If incoming is a input layer, which will only happen when
            % there's a model inside of a sequential, skip the input layer.
            %SourceIdx = findLayerSpecIdx(LayerSpecs{L}.InConns{1}.FromLayerName, LayerSpecs);
            if isequal(LayerSpecs{SourceIdx}.Type, 'InputLayer') && ~isempty(LayerSpecs{SourceIdx}.InConns)
                [NNTSourceName, NNTSourceNum] = findInConnSource(LayerSpecs{SourceIdx}.InConns{1}, LayerSpecs, NNTLayerGroups);
            end
            if LayerSpecs{SourceIdx}.NumOutputs > 1 
                NNTSourceString = sprintf('%s/out%d', NNTSourceName, NNTSourceNum);
            else 
                NNTSourceString = NNTSourceName;
            end
            NNTTargetString = sprintf('%s', NNTLayerGroups{L}{1}.Name);
            if strcmp(LayerSpecs{SourceIdx}.Type, 'TimeDistributedIn')
                NNTSourceString = [NNTSourceString '/out'];
            end
            if isa(NNTLayerGroups{L}{1}, 'nnet.cnn.layer.SequenceUnfoldingLayer')
                NNTTargetString = [NNTTargetString '/in'];
            end
            NNTLayerGraph = connectLayers(NNTLayerGraph, NNTSourceString, NNTTargetString);
        else
            for i = 1:numel(LayerSpecs{L}.InConns)
                % Connect the source of this InConn to layer L.
                [NNTSourceName, NNTSourceNum, SourceIdx] = findInConnSource(LayerSpecs{L}.InConns{i}, LayerSpecs, NNTLayerGroups);
                if LayerSpecs{SourceIdx}.NumOutputs > 1 
                    NNTSourceString = sprintf('%s/out%d', NNTSourceName, NNTSourceNum);
                else 
                    NNTSourceString = NNTSourceName;
                end 
                NNTTargetString = sprintf('%s/in%d', NNTLayerGroups{L}{1}.Name, i);
                NNTLayerGraph = connectLayers(NNTLayerGraph, NNTSourceString, NNTTargetString);
            end
        end
    end
    if isequal(LayerSpecs{L}.Type, 'TimeDistributedIn')
        name = LayerSpecs{L}.Name(1:length(LayerSpecs{L}.Name)-2);
        NNTLayerGraph = connectLayers(NNTLayerGraph, [name 'in/miniBatchSize'], [name 'out/miniBatchSize']);
    end
end
end

function NNTLayerGraph = removeDisconnectedLayers(NNTLayerGraph)
DisconnectedLayerNums = setdiff(1:numel(NNTLayerGraph.Layers), unique(NNTLayerGraph.HiddenConnections.EndNodes(:)));
if any(DisconnectedLayerNums)
    NNTLayerGraph = removeLayers(NNTLayerGraph, {NNTLayerGraph.Layers(DisconnectedLayerNums).Name});
end
end

function n = nntLayerNum(LayerName, NNTLayerGraph)
[~,n] = ismember(LayerName, {NNTLayerGraph.Layers.Name});
end

function [NNTSourceName, NNTSourceNum, SourceIdx] = findInConnSource(InConn, LayerSpecs, NNTLayerGroups)
% InConn is a Tensor. Find the name and output number of the NNT layer that
% sends this connection. Along the way, ignore layers that expand to nothing.
SourceIdx = findLayerSpecIdx(InConn.FromLayerName, LayerSpecs);
if isempty(SourceIdx)
    re = regexp(InConn.FromLayerName, '^tf_op_layer_', 'end'); 
    if ~isempty(re) 
        SourceIdx = findLayerSpecIdx(InConn.FromLayerName(re+1:end), LayerSpecs);
    end 
end
switch numel(NNTLayerGroups{SourceIdx})
    case 0
        % The source LayerSpec expanded into no NNT layers. Make a
        % recursive call on the source LayerSpec's InConn{1}.
        [NNTSourceName, NNTSourceNum] = findInConnSource(LayerSpecs{SourceIdx}.InConns{1}, LayerSpecs, NNTLayerGroups);
    case 1
        % The source LayerSpec expanded into a single NNT layer. Return that
        % NNT layer and the requested output number.
        NNTSourceName = NNTLayerGroups{SourceIdx}{1}.Name;
        NNTSourceNum = InConn.FromOutputNum;
    otherwise
        % The source LayerSpec expanded into multiple NNT layers. Return
        % the last NNT layer and output number 1.
        if isa(NNTLayerGroups{SourceIdx}{end}, 'nnet.cnn.layer.ClassificationOutputLayer') || ...
                isa(NNTLayerGroups{SourceIdx}{end}, 'nnet.cnn.layer.RegressionOutputLayer') || ...
                isa(NNTLayerGroups{SourceIdx}{end},'nnet.keras.layer.BinaryCrossEntropyRegressionLayer')
            % If a LayerSpec is an output and expands to have an output
            % layer, the second to last layer in the group should be the 
            % source layer.
            NNTSourceName = NNTLayerGroups{SourceIdx}{end - 1}.Name;
        else 
            NNTSourceName = NNTLayerGroups{SourceIdx}{end}.Name;
        end 
        NNTSourceNum = 1;
end
end

function SourceIdx = findLayerSpecIdx(LayerName, LayerSpecs)
    SourceIdx = find(cellfun(@(Spec)isequal(Spec.Name, LayerName), LayerSpecs));
end
