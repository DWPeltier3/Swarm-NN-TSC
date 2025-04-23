function y = tfTensorListFromTensor(tensor, ~)

%   Copyright 2023 The MathWorks, Inc.

    % The TensorListFromTensor operator returns a TensorList tensor:
    %   tf.Tensor(<TensorList>, shape=(), dtype=variant)
    % 
    % Hence, tfTensorListFromTensor returns an array of structs, each holding a 
    % dlarray and its rank.
    % 
    % element_shape: The only valid scalar shape tensor is the fully unknown 
    % shape specified as -1 can only be -1 if it is a scalar else it has to
    % be a rank 1 tensor containing the shape of the tensor elements to be 
    % reserved, hence this will always be in forward TF format
    % tensor_shape = element_shape.value; 
    % tensor_shape_rank = element_shape.rank;

    tensorVal = tensor.value;
    tensorRank = tensor.rank;
    
    tensorVal = permute(tensorVal, tensorRank:-1:1); 

    % tval should be in forward TF format
    numElements = size(tensorVal,1);
    elemRank = tensorRank - 1;
    
    if tensorRank > 1    
        elemShape = size(tensorVal, 2:tensorRank);
    else        
        elemShape = 1;
    end

    y = [];
    for i = 1:numElements        
        if elemRank <= 1
            % Keep forward TF format for individual element tensors
            elemTensor = dlarray(tensorVal(i,:));
            element = struct('value', elemTensor, 'rank', elemRank);            
        else
            % Reverse TF format as element tensor rank is greater than 1
            elemTensor = tensorVal(i,:);
            elemTensor = reshape(elemTensor, elemShape);
            elemTensor = permute(elemTensor, elemRank:-1:1);
            element = struct('value', elemTensor, 'rank', elemRank);            
        end    
        y = [y element]; %#ok<AGROW>
    end
end






