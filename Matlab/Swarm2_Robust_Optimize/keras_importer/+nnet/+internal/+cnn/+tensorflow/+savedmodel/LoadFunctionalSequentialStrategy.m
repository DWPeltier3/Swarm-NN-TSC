classdef LoadFunctionalSequentialStrategy < nnet.internal.cnn.tensorflow.savedmodel.LoadTFObjectStrategy
% Strategy class to import a functional or sequential model into a
% layergraph or dlnetwork object. This object should include an 
% ObjectNode property which will be used in translation.

%   Copyright 2022-2023 The MathWorks, Inc.
    
    methods
        function preProcessTranslatorObject(this, ObjectLoaderResult, InternalTrackableGraph, GraphDef, ~, ~, ~)
            import nnet.internal.cnn.keras.util.*;
            % Don't need any pre-processing for Functional or Sequential models
            if strcmp(this.ObjectNode.user_object.identifier, '_tf_keras_sequential')
                ObjectLoaderResult.APIType = 'Sequential'; 
            else
                ObjectLoaderResult.APIType = 'Functional'; 
            end
            % Find FunctionDef for this model, to help with node contraction 
            fcnIdx = this.getChildWithName('call_and_return_all_conditional_losses'); 
            if isempty(fcnIdx)
                fcnIdx = this.getChildWithName('__call__'); 
            end

            if isempty(fcnIdx)
                objMetaData = this.ObjectNode.user_object.metadata;
                objMetaDataStruct = jsondecode(objMetaData);
                objName = '';
                if isfield(objMetaDataStruct,'name')
                    objName = objMetaDataStruct.name;
                end                
                this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:NoGraphFoundForObject', MessageArgs={objName});
                ObjectLoaderResult.HasFcn = false; 
                return;
            end

            fcnNode = InternalTrackableGraph.NodeStruct{fcnIdx}; 
            try
                fcnName = fcnNode.function.concrete_functions{1}; 
            catch
                return;
            end 
            ObjectLoaderResult.HasFcn = true; 
            FunctionDef = GraphDef.findFunction(fcnName); 
            ObjectLoaderResult.FunctionDef = FunctionDef; 
            ObjectLoaderResult.TranslationStrategy = this;
        end

        function translateObject(this, ObjectLoaderResult, InternalTrackableGraph, GraphDef, SavedModelPath)

            % disable Keras Importer warnings about unsupported layer
            % settings
            warnState = warning('off','nnet_cnn_kerasimporter:keras_importer:UnsupportedLayerSettingsWarning');
            warning('off','nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer');
            warning('off','nnet_cnn_kerasimporter:keras_importer:BatchNormNegVar');
            % Ensure that the warning state is restored upon function completion
            C = onCleanup(@()warning(warnState));
            ModelConfig = jsondecode(this.ObjectNode.user_object.metadata); 
            ModelConfig = nnet.internal.cnn.tensorflow.savedmodel.util.iMakeModelConfigKICompatible(ModelConfig, InternalTrackableGraph, []);
            KM = nnet.internal.cnn.keras.ParsedKerasModel(ModelConfig, [], true);
            AM = nnet.internal.cnn.keras.AssembledModel(KM);
            AM.WeightsImported = true; 
                
            % import weights always
            checkpointindexpath = [fullfile(SavedModelPath, 'variables') filesep 'variables.index']; 
            AM = this.translateWeightsFromAM(AM, checkpointindexpath, InternalTrackableGraph);

            LayersOrGraph = translateAssembledModel(AM, [], AM.WeightsImported, false, []);
            % re-enable Keras Importer warnings about unsupported layer settings
            warning('on','nnet_cnn_kerasimporter:keras_importer:BatchNormNegVar');
            warning('on','nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer');
            warning('on','nnet_cnn_kerasimporter:keras_importer:UnsupportedLayerSettingsWarning');
            
            if isa(LayersOrGraph,'nnet.cnn.layer.Layer')
                lg = layerGraph(LayersOrGraph);
            else
                lg = LayersOrGraph;
            end

            [lg, hasUnsuppOp] = nnet.internal.cnn.tensorflow.gcl.translateTFKerasLayersByName_v2(InternalTrackableGraph, GraphDef, lg, SavedModelPath, this.ImportManager, AM.LayerSpecs);

            ObjectLoaderResult.LayerToOutName = containers.Map;
            
            if isa(lg,'nnet.cnn.LayerGraph')
                layers = lg.Layers;
            else
                layers = lg;
            end

            for i = 1:numel(layers)
                ObjectLoaderResult.LayerToOutName(layers(i).Name) = layers(i).OutputNames;
            end

            if ~hasUnsuppOp
                dlnetOrNone = dlnetwork(lg, 'Initialize', false);
            else
                dlnetOrNone = dlnetwork.empty;
            end

            ObjectLoaderResult.Instance = dlnetOrNone;
            ObjectLoaderResult.Class = ModelConfig.name; 
            ObjectLoaderResult.TranslationStrategy = this;
            %Store layer names to ouput names map here
        end
        
        function AM = translateWeightsFromAM(~, AM, checkpointindexpath, InternalTrackableGraph)
            % Manually adds weight structures to the AssembledModel
            % LayerSpecs
            import nnet.internal.cnn.tensorflow.*;
            numLayers = numel(AM.LayerSpecs);
            for i = 1:numLayers
                layerName = AM.LayerSpecs{i}.Name; 
                
                % Get weights specially if the network is a Bidirectional
                % LSTM 
                if strcmp(AM.LayerSpecs{i}.Type, 'Bidirectional')
                    layerWeights = AM.LayerSpecs{i}.Translator.WeightNames; 
                    for j = 1:numel(layerWeights)
                        weightIdx = InternalTrackableGraph.getBidirectionalWeightNames(layerName, layerWeights{j}); 
                        [curWeight, ~] = tf2mex('checkpoint', checkpointindexpath, weightIdx.forward); 
                        curWeightName = ['forward_lstm_' layerWeights{j}];
                        AM.LayerSpecs{i}.Weights.(curWeightName) = curWeight;

                        [curWeight, ~] = tf2mex('checkpoint', checkpointindexpath, weightIdx.backward); 
                        curWeightName = ['backward_lstm_' layerWeights{j}]; 
                        AM.LayerSpecs{i}.Weights.(curWeightName) = curWeight;
                    end 
                else
                    % Normal case for layers. 
                    if isempty(AM.LayerSpecs{i}.Translator.WeightNames) && ~(AM.LayerSpecs{i}.Translator.KerasLayerType == "Dense")
                        % Do not gather weights for empty layers except if
                        % it is a Dense placeholder layer
                        continue; 
                    end
                    
                    nodeIdx = InternalTrackableGraph.LayerSpecToNodeIdx(layerName); 
                    variablesIdx = InternalTrackableGraph.getChildWithName(nodeIdx, 'variables'); 
                    if ~isempty(variablesIdx)
                        variables = InternalTrackableGraph.NodeStruct{variablesIdx}.children; 
                    else 
                        variables = []; 
                    end 
                    
                    for j = 1:numel(variables)
                        curVarIdx = variables(j).node_id; 
                        curVar = InternalTrackableGraph.NodeStruct{curVarIdx + 1};
                        [~, curWeightName] = fileparts(curVar.variable.name); 
                        [curWeight, ~] = tf2mex('checkpoint', checkpointindexpath, curVarIdx); 
                        AM.LayerSpecs{i}.Weights.(curWeightName) = curWeight;
                    end
                end
            end
        end
        end
end