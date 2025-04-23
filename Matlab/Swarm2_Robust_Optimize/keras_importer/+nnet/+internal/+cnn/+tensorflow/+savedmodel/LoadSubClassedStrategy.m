classdef LoadSubClassedStrategy < nnet.internal.cnn.tensorflow.savedmodel.LoadTFObjectStrategy
%   Copyright 2022-2023 The MathWorks, Inc.

    methods 

        function preProcessTranslatorObject(this, ObjectLoaderResult, InternalTrackableGraph, GraphDef, SavedModelPath, packageName, modelPath)
            import nnet.internal.cnn.keras.util.*;
            ObjectLoaderResult.APIType = 'SubClassedModel'; 
            ObjectLoaderResult.Instance = dlnetwork.empty;

            fcnIdx = this.getChildWithName('call_and_return_all_conditional_losses'); 
            if isempty(fcnIdx)
                fcnIdx = this.getChildWithName('__call__'); 
            end

            if isempty(fcnIdx)
                objMetaData = this.ObjectNode.user_object.metadata;
                % Decode Keras metadata if available
                if ~isempty(objMetaData)
    		        objMetaDataStruct = jsondecode(objMetaData);
	                objName = '';
                    if isfield(objMetaDataStruct,'name')
                    	objName = objMetaDataStruct.name;
                    end
                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:NoGraphFoundForObject', MessageArgs={objName});                    
                end
                ObjectLoaderResult.HasFcn = false;
                return;
            end

            fcnNode = InternalTrackableGraph.NodeStruct{fcnIdx}; 
            try
                if numel(fcnNode.function.concrete_functions) > 1
                    concrete_fcns = fcnNode.function.concrete_functions;
                    num_nodes = getNumNodesInFunction(concrete_fcns, GraphDef);
                    [~,minidx] = min(num_nodes);
                    fcnName = fcnNode.function.concrete_functions{minidx}; 
                else
                    fcnName = fcnNode.function.concrete_functions{1}; 
                end         
            catch
                return;
            end 
            ObjectLoaderResult.HasFcn = true; 
            FunctionDef = GraphDef.findFunction(fcnName); 
            ObjectLoaderResult.FunctionDef = FunctionDef; 
            TranslatedChildren = nnet.internal.cnn.tensorflow.savedmodel.collectUnusedChildrenObjects(ObjectLoaderResult);
            try 
                model_config = jsondecode(this.ObjectNode.metadata); 
                ObjectLoaderResult.Class = model_config.class_name;
            catch
                ObjectLoaderResult.Class = ['k' FunctionDef.Signature.legalname];
            end
            
            % Setup code generation context. 
            codegenerator = ...
                nnet.internal.cnn.tensorflow.gcl.SubClassedModelTranslator_v2(...
                    SavedModelPath, ...
                    InternalTrackableGraph, ...
                    GraphDef, ...
                    FunctionDef, ...
                    TranslatedChildren, ...
                    ObjectLoaderResult.Class, ...
                    this.ObjectNode, ... 
                    ObjectLoaderResult.Namespace, ...
                    packageName, ...
                    modelPath, ...
                    this.ImportManager ...
                );
            % pre-process FunctionGraph
            codegenerator.computeUsedInputsOutputsForChildren(ObjectLoaderResult);
            codegenerator.contractAllModels(); 

            % Store the Codegenerator for later use
            ObjectLoaderResult.CodegeneratorObject = codegenerator;   
            ObjectLoaderResult.TranslationStrategy = this;
        end
        
        function translateObject(~, ObjectLoaderResult, ~, ~, ~)
            % convert GraphDef to match pre-processed digraph 
            TranslatedChildren = nnet.internal.cnn.tensorflow.savedmodel.collectUnusedChildrenObjects(ObjectLoaderResult);
            codegenerator = ObjectLoaderResult.CodegeneratorObject;

            codegenerator.convertGraphDef(); 
            
            % Update subclassed model translator obj with Top level layer info
            codegenerator.IsTopLevelModel = ObjectLoaderResult.IsTopLevelLayer;

            % Write out subclassed model to disk.
            codegenerator.writeSubClassedModelWrapper(); 
            
            try 
                ObjectLoaderResult.Instance = codegenerator.instantiateSubClassedModelWrapper(TranslatedChildren,ObjectLoaderResult.LayerServingDefaultOutputs); 
            catch 
                ObjectLoaderResult.Instance = []; 
            end 
        end

    end
end

function num_nodes = getNumNodesInFunction(fcnCells, GraphDef)
    num_nodes = zeros(1,numel(fcnCells));
    for i = 1:numel(fcnCells)
            fcnCell = fcnCells{i};
            FunctionDef = GraphDef.findFunction(fcnCell);
            num_nodes(i) = numel(FunctionDef.node_def);
    end
end