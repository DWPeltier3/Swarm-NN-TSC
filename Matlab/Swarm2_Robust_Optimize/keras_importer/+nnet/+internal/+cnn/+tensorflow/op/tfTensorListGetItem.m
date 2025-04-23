function y = tfTensorListGetItem(inputHandle, index, elementShape)

%   Copyright 2023 The MathWorks, Inc.

    % The TensorListGetItem operator returns a Tensor at an index in a TensorList tensor:
    %   tf.Tensor(<TensorList>, shape=(), dtype=variant)
    % 
    % Hence, tfTensorListGetItem returns a Tensor
    % 
    % element_shape: The only valid scalar shape tensor is the fully unknown 
    % shape specified as -1 

    indexVal = index.value + 1;
    yVal = inputHandle(indexVal).value;
    yRank = inputHandle(indexVal).rank;
    
    if yRank == -1
        % input_handle is an unitialized TensorList
        elementShape = elementShape.value; 
        yRank = numel(elementShape);
        if yRank <= 1
            % Forward TF format
            yVal = dlarray(zeros(elementShape));            
        else
            % Reverse TF format as rank > 1
            yVal = dlarray(zeros([flip(elementShape)]));            
        end      
    end
    y = struct('value', yVal, 'rank', yRank);    
end

