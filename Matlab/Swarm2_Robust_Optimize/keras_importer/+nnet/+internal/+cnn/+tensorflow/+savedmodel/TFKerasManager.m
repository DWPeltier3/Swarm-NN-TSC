classdef TFKerasManager < handle
    %TFKERASMANAGER Manages a SavedModel's object graph which stores
    %the model's Keras object structure.

%   Copyright 2020-2023 The MathWorks, Inc.
    
    properties
        % For general use 
        SavedModelPath
        RootClassType
        InternalTrackableGraph % Tracks all of the node dependencies
        
        % For Sequential and function APIs 
        AM % AssembledModel
        TrainingConfig
        LayerGraph
        
    end
    
    methods
        function obj = TFKerasManager(object_graph_def, path, importManager, importNetwork)
            %TFKERASMANAGER Construct a Keras Manager from the saved model
            %object graph def message. 
            import nnet.internal.cnn.tensorflow.*;
			import nnet.internal.cnn.keras.util.*;
            
            % disable Keras Importer warnings about unsupported layer
            % settings
            warnState(1) = warning('off','nnet_cnn_kerasimporter:keras_importer:UnsupportedLayerSettingsWarning');
            warnState(2) = warning('off','nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer');
            warnState(3) = warning('off','nnet_cnn_kerasimporter:keras_importer:BatchNormNegVar');
            warnState(4) = warning('off','nnet_cnn_kerasimporter:keras_importer:UnsupportedPReLUParameterSize');

            % Ensure that the warning state is restored upon function completion
            C = onCleanup(@()warning(warnState));
            
            obj.InternalTrackableGraph = savedmodel.InternalTrackableGraph(object_graph_def, importManager); 
            warning('on','nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer');

            obj.SavedModelPath = path;
            ModelObject = object_graph_def.nodes{1}.user_object; % 1 corresponds to the root trackable object. 
            obj.RootClassType = ModelObject.identifier; 
            if strcmp(ModelObject.identifier , '_tf_keras_model') ||...
                    strcmp(ModelObject.identifier, '_tf_keras_sequential') || ...
                    strcmp(ModelObject.identifier, '_tf_keras_network')
                % Decode metadata if available
                if ~isempty(ModelObject.metadata)
                    ModelConfig = jsondecode(ModelObject.metadata);
                else
                    obj.LayerGraph = []; 
                    return; 
                end

                
                if ~(strcmp(ModelConfig.class_name, 'Model') || strcmp(ModelConfig.class_name, 'Sequential') || strcmp(ModelConfig.class_name, 'Functional'))
                    obj.LayerGraph = []; 
                    return; 
                end 
                
                if strcmp(ModelConfig.class_name, 'Functional')
                    % We may need to change Keras Importer rather than
                    % change data in the future. 
                    ModelConfig.class_name = 'Model'; 
                end 
                    
                if strcmp(importManager.TargetNetwork, 'dagnetwork')
                    % Configuring output layer only necessary if the target
                    % is a dagnetwork. 
                    if isfield(ModelConfig, 'training_config') && ~isempty(ModelConfig.training_config.loss)
                        obj.TrainingConfig = ModelConfig.training_config;
                    elseif ~isempty(importManager.OutputLayerType)
                        obj.TrainingConfig = struct; 
                        if isa(importManager.OutputLayerType, 'string') || isa(importManager.OutputLayerType, 'char')
                            switch importManager.OutputLayerType
                                case {'classification', 'pixelclassification'}
                                    obj.TrainingConfig.loss = 'categorical_crossentropy';
                                case 'regression'
                                    obj.TrainingConfig.loss = 'mse';
                                case 'binarycrossentropyregression'
                                    obj.TrainingConfig.loss = 'binary_crossentropy';
                                otherwise
                                    assert(false);
                            end
                        end
                    else
                        if importNetwork
                            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NetworkOutputLayerTypeMissing')));
                        else
                            importManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:LayersOutputLayerTypeMissing');                            
                        end
                    end
                else 
                    obj.TrainingConfig = []; 
                end
                  
                % make tf.keras model config compatible with keras importer
                ModelConfig = savedmodel.util.iMakeModelConfigKICompatible(ModelConfig, obj.InternalTrackableGraph, '');
                isTensorFlowModel = true;
                KM = nnet.internal.cnn.keras.ParsedKerasModel(ModelConfig, obj.TrainingConfig, isTensorFlowModel); 

                % Convert TimeDistributed custom-layer into
                % sequenceFolding/unFolding layers only for DAGNetworks
                if strcmp(importManager.TargetNetwork, 'dagnetwork')
                    % Convert TimeDistributed custom-layer into
                    % sequenceFolding/unFolding layers only for DAGNetworks
                    timeDistributedCustomLayersIdx = find(cellfun(@(layer) isa(layer.Config, 'nnet.internal.cnn.keras.KerasBaseLevelLayerConfig') && ...
                        strcmp(layer.ClassName, 'TimeDistributed'), KM.Model.Config.Layers));
                    for i = 1:numel(timeDistributedCustomLayersIdx)
                        currentLayerConfig = KM.Model.Config.Layers{timeDistributedCustomLayersIdx(i)}.Config;
                        updatedLayerConfig = nnet.internal.cnn.keras.KerasTimeDistributedModelConfig(currentLayerConfig.KerasStruct, isTensorFlowModel);
                        KM.Model.Config.Layers{timeDistributedCustomLayersIdx(i)}.Config = updatedLayerConfig;
                    end
                end
                
                obj.AM = nnet.internal.cnn.keras.AssembledModel(KM);                
                obj.AM.WeightsImported = true; 
                
                % import weights always
                checkpointindexpath = [fullfile(obj.SavedModelPath, 'variables') filesep 'variables.index']; 
                obj.translateWeightsFromAM(checkpointindexpath);
                
                % 'ImageInputSize' does not apply to MI networks.
                ImageInputSize = importManager.ImageInputSize;
                if strcmp(obj.AM.ModelType, 'DAG') && numel(obj.AM.InputLayerIndices) > 1 && ~isempty(ImageInputSize) 
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MIImageInputSize'))); 
                end
                LayersOrGraph = translateAssembledModel(obj.AM, obj.TrainingConfig, obj.AM.WeightsImported, false, ImageInputSize);
                
                % enable Keras Importer warnings about unsupported layer
                % settings
                warning('on','nnet_cnn_kerasimporter:keras_importer:UnsupportedPReLUParameterSize');
                warning('on','nnet_cnn_kerasimporter:keras_importer:BatchNormNegVar');
                warning('on','nnet_cnn_kerasimporter:keras_importer:UnsupportedLayerSettingsWarning');
                if isa(LayersOrGraph,'nnet.cnn.layer.Layer')
                    obj.LayerGraph = layerGraph(LayersOrGraph);
                else
                    obj.LayerGraph = LayersOrGraph;
                end
            else
                % Assume that model is created using Subclassing
                obj.LayerGraph = [];
            end
        end
    end
    
    methods (Access=private)
        function translateWeightsFromAM(this, checkpointindexpath)
            % Manually adds weight structures to the AssembledModel
            % LayerSpecs
            import nnet.internal.cnn.tensorflow.*;
            numLayers = numel(this.AM.LayerSpecs);
            for i = 1:numLayers
                layerName = this.AM.LayerSpecs{i}.Name; 
                
                % Get weights specially if the network is a Bidirectional
                % LSTM 
                if strcmp(this.AM.LayerSpecs{i}.Type, 'Bidirectional')
                    layerWeights = this.AM.LayerSpecs{i}.Translator.WeightNames; 
                    for j = 1:numel(layerWeights)
                        weightIdx = this.InternalTrackableGraph.getBidirectionalWeightNames(layerName, layerWeights{j}); 
                        [curWeight, ~] = tf2mex('checkpoint', checkpointindexpath, weightIdx.forward); 
                        curWeightName = ['forward_lstm_' layerWeights{j}];
                        this.AM.LayerSpecs{i}.Weights.(curWeightName) = curWeight;

                        [curWeight, ~] = tf2mex('checkpoint', checkpointindexpath, weightIdx.backward); 
                        curWeightName = ['backward_lstm_' layerWeights{j}]; 
                        this.AM.LayerSpecs{i}.Weights.(curWeightName) = curWeight;
                    end 
                else
                    % Normal case for layers. 
                    if isempty(this.AM.LayerSpecs{i}.Translator.WeightNames) && ~(this.AM.LayerSpecs{i}.Translator.KerasLayerType == "Dense")
                        % Do not gather weights for empty layers except if
                        % it is a Dense placeholder layer
                        continue; 
                    end
                    
                    nodeIdx = this.InternalTrackableGraph.LayerSpecToNodeIdx(layerName); 
                    variablesIdx = this.InternalTrackableGraph.getChildWithName(nodeIdx, 'variables'); 
                    if ~isempty(variablesIdx)
                        variables = this.InternalTrackableGraph.NodeStruct{variablesIdx}.children; 
                    else 
                        variables = []; 
                    end 
                    
                    for j = 1:numel(variables)
                        curVarIdx = variables(j).node_id; 
                        curVar = this.InternalTrackableGraph.NodeStruct{curVarIdx + 1};
                        [~, curWeightName] = fileparts(curVar.variable.name); 
                        [curWeight, ~] = tf2mex('checkpoint', checkpointindexpath, curVarIdx); 
                        this.AM.LayerSpecs{i}.Weights.(curWeightName) = curWeight;
                    end
                end
            end
        end
    end
end
