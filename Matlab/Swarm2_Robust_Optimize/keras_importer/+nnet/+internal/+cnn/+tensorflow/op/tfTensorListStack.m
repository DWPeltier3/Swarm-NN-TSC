function y = tfTensorListStack(tensorList, elementShapeVal)
    %{{import_statement}}

%   Copyright 2023 The MathWorks, Inc.

    % The TensorListStack operator takes a tensor_list as its first input:
    %   tf.Tensor(<TensorList>, shape=(), dtype=variant)
    % 
    % element_shape: The only valid scalar shape tensor is the fully unknown 
    % shape specified as -1 can only be -1 if it is a scalar else it has to
    % be a rank 1 tensor containing the shape of the tensor elements to be 
    % reserved, hence this will always be in forward TF format
    if isempty(tensorList)
        y = struct('value', [], 'rank', 0);
        return;
    end
    
    % tensor_list should have atleast one element
    numElements = numel(tensorList);
    tensorListRank = tensorList(1).rank;
    elementShapeVal = elementShapeVal.value; 
    
    if tensorListRank ~= -1    
        y = tfPack(0, tensorList(:)); 
    else
        tensorRank = numel(elementShapeVal) + 1;
        if tensorRank <= 1
            % Forward TF format
            yVal = dlarray(zeros([numElements, elementShapeVal]));            
        else
            % Reverse TF format as rank > 1
            yVal = dlarray(zeros([flip(elementShapeVal), numElements]));            
        end      
        y = struct('value', yVal, 'rank', tensorRank);
    end   
end






