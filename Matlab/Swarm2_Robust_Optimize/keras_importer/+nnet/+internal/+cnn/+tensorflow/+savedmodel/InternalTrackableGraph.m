classdef InternalTrackableGraph < handle
    % INTERNALTRACKABLEGRAPH Tensorflow representation of an InternalTrackableGraph
    % Used as a utililty to lookup the contents of a Keras layer and its
    % dependency path to the root. This is useful to lookup specific
    % tensornames

%   Copyright 2020-2022 The MathWorks, Inc.
    
    properties
        DependencyGraph
        NodeStruct
        ConcreteFunctions
        
        LayerSpecToNodeIdx
        LayerSpecIdxToNodeIdx 
    end 
    
    methods (Access = public)
        function obj = InternalTrackableGraph(obj_graph_def, importManager)
            obj.NodeStruct = obj_graph_def.nodes; 
            obj.ConcreteFunctions = obj_graph_def.concrete_functions; 
            obj.DependencyGraph = digraph(); 
            
            for i = 1:numel(obj_graph_def.nodes)
                obj.DependencyGraph = obj.DependencyGraph.addnode(num2str(i)); 
            end 
            
            for i = 1:numel(obj_graph_def.nodes)
                curnode = obj_graph_def.nodes{i};
                for j = 1:numel(curnode.children)
                    obj.DependencyGraph = ... 
                        obj.DependencyGraph.addedge(i, (curnode.children(j).node_id) + 1); 
                end
            end
            
            obj.LayerSpecToNodeIdx = containers.Map();
            obj.LayerSpecIdxToNodeIdx = []; 
            
            for i = 1:numel(obj_graph_def.nodes)
                if isfield(obj.NodeStruct{i}, 'user_object') && ...
                        (strcmp(obj.NodeStruct{i}.user_object.identifier, '_tf_keras_layer') || ...
                         strcmp(obj.NodeStruct{i}.user_object.identifier, '_tf_keras_input_layer') || ...
                         strcmp(obj.NodeStruct{i}.user_object.identifier, '_tf_keras_rnn_layer') || ...
                         strcmp(obj.NodeStruct{i}.user_object.identifier, '_tf_keras_network') ||... 
                         strcmp(obj.NodeStruct{i}.user_object.identifier, '_tf_keras_sequential') ||...
                        strcmp(obj.NodeStruct{i}.user_object.identifier, '_tf_keras_model'))
                    
                    try
                        KerasLayerConfig = jsondecode(obj.NodeStruct{i}.user_object.metadata);
                        
                        if isfield(KerasLayerConfig, 'config') 
                            obj.LayerSpecToNodeIdx(KerasLayerConfig.name) = i; 
                            obj.LayerSpecIdxToNodeIdx(end + 1) = i;  
                        end
                    catch
                        importManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:TFGraphMissingKerasMetadata',MessageArgs={i});                                                
                    end
                end
            end
        end
                
        function weightIdx = getBidirectionalWeightNames(this, layerName, weightName)
            % returns the Bidirectional LSTM weight indices in a struct
            % organized by forward/backward. 
            if isKey(this.LayerSpecToNodeIdx, layerName)
                nodeStructIdx = this.LayerSpecToNodeIdx(layerName); 
                
                % Get the forward variable name
                forwardIdx = this.getChildWithName(nodeStructIdx, 'forward_layer');
                forwardParamsIdx = this.getChildWithName(forwardIdx, 'variables');
                for i = 1:numel(this.NodeStruct{forwardParamsIdx}.children)
                    curForwardVariableIdx = this.NodeStruct{forwardParamsIdx}.children(i).node_id + 1; 
                    
                    curForwardVariable = this.NodeStruct{curForwardVariableIdx}.variable.name; 
                    curForwardVariableParts = strsplit(curForwardVariable, '/'); 
                    if strcmp(curForwardVariableParts{end}, weightName)
                        % tf2mex expects 0 based index 
                        weightIdx.forward = curForwardVariableIdx - 1; 
                        break; 
                    end
                end
                
                % Get the backward Variable name. 
                backwardIdx = this.getChildWithName(nodeStructIdx, 'backward_layer');
                backwardParamsIdx = this.getChildWithName(backwardIdx, 'variables');
                for i = 1:numel(this.NodeStruct{backwardParamsIdx}.children)
                    curBackwardVariableIdx = this.NodeStruct{backwardParamsIdx}.children(i).node_id + 1; 
                    curBackwardVariable = this.NodeStruct{curBackwardVariableIdx}.variable.name;
                    curBackwardVariableParts = strsplit(curBackwardVariable, '/'); 
                    if strcmp(curBackwardVariableParts{end}, weightName)
                        % tf2mex expects 0 based index 
                        weightIdx.backward = curBackwardVariableIdx - 1; 
                        break; 
                    end
                end
            else
                weightIdx = []; 
            end
        end 
                        
        function childidx = getChildWithName(this, nodeStructIdx, name)
            curNode = this.NodeStruct{nodeStructIdx};
            for i = 1:numel(curNode.children)
                if strcmp(curNode.children(i).local_name, name)
                    childidx = curNode.children(i).node_id + 1;
                    return; 
                end
            end 
            % no child with name discovered. 
            childidx = [];     
        end
        
        function namespace = getObjectNamespace(this, nodeidx)
            root_node_idx = 1; 
            namespace = this.DependencyGraph.shortestpath(num2str(root_node_idx), num2str(nodeidx)); 
            for i = 1:numel(namespace) 
                curObj = this.NodeStruct{str2double(namespace{i})}.user_object; 
                if isempty(curObj.metadata) 
                    continue; 
                end
                curObj = jsondecode(curObj.metadata); 
                namespace{i} = curObj.name; 
            end
            
            if numel(namespace) > 1
                namespace = strjoin(namespace(2:end), '/');
            else 
                namespace = ''; 
            end 
        end
        
    end
end 
