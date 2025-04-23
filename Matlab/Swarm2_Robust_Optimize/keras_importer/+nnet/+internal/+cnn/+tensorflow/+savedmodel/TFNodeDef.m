classdef TFNodeDef
    % TFNodeDef Class representation of a TensorFlow graph node.

%   Copyright 2020-2021 The MathWorks, Inc.

    properties
        name 
        op 
        input 
        device 
        attr 
        
        MATLABIdentifierName
        ParentFcnName
    end 
    
    methods
        function obj = TFNodeDef(node_def) 
            obj.name = node_def.name; 
            obj.op = node_def.op; 
            obj.input = node_def.input;
            obj.device = node_def.device; 
            obj.attr = node_def.attr; 
        end
    end
end
