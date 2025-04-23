classdef TFGraphInfo
    % TFGraphInfo Class representation of TensorFlow 2.0 computational graph
    % metadata. 

%   Copyright 2020-2021 The MathWorks, Inc.
    
    properties
        MetaGraphVersion
        StrippedOpList
        Tags
        TensorFlowVersion
        TensorFlowGitVersion
        StrippedDefaultAttrs
        FunctionAliases
    end
    
    methods
        function obj = TFGraphInfo(meta_info_def)
            obj.MetaGraphVersion     = meta_info_def.meta_graph_version;
            obj.StrippedOpList       = meta_info_def.stripped_op_list; 
            obj.Tags                 = meta_info_def.tags; 
            obj.TensorFlowVersion    = meta_info_def.tensorflow_version; 
            obj.TensorFlowGitVersion = meta_info_def.tensorflow_git_version;
            obj.StrippedDefaultAttrs = meta_info_def.stripped_default_attrs; 
            obj.FunctionAliases      = meta_info_def.function_aliases; 
        end
    end
end

