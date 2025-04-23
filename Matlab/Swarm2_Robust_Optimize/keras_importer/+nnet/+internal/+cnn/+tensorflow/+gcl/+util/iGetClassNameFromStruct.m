function className = iGetClassNameFromStruct(nodestruct)
    % This function retrieves the class_name of a object struct.
    % This corresponds to the original python class name.

%   Copyright 2020-2021 The MathWorks, Inc.

    config = jsondecode(nodestruct.user_object.metadata); 
    if isfield(config, 'class_name')
        className = config.class_name; 
    else 
        className = '';
    end 
end 
