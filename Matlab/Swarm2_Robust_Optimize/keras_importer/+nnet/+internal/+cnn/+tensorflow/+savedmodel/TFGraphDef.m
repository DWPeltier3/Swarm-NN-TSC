classdef TFGraphDef < handle 
    %TFGraphDef Class representation of the main TensorFlow graph (stored in Nodes) 
    % and all model subgraphs (stored in FunctionLibrary).

%   Copyright 2020-2021 The MathWorks, Inc.
    
    properties
        Nodes
        FunctionLibrary
        Version
    end
        
    methods
        function obj = TFGraphDef(graph_def)
            import nnet.internal.cnn.tensorflow.*;
            obj.Nodes = savedmodel.TFNodeDef.empty(); 
            for i = 1:numel(graph_def.node)
                obj.Nodes(i) = savedmodel.TFNodeDef(graph_def.node(i)); 
            end 

            for i = 1:numel(graph_def.library.function)
                obj.FunctionLibrary{i} = savedmodel.TFFunction(graph_def.library.function(i)); 
            end 
            obj.Version = {graph_def.versions, graph_def.version}; 
        end
                     
        function fcn = findFunction(this, name)
            try
                funcidx = cellfun(@(fcn)(strcmp(fcn.Signature.name, name)), this.FunctionLibrary);
                fcn = this.FunctionLibrary{funcidx};
            catch
                fcn = []; 
            end
        end
        
        function node = findNodeByName(this, name)
            node = [];
            for i = numel(this.Nodes):-1:1
                if strcmp(this.Nodes(i).name,name)
                    node = this.Nodes(i);                
                    return;
                end
            end 
        end
        
    end
end

