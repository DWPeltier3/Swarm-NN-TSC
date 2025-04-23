function y = tfTensorListReserve(elementShape, numElements)

%   Copyright 2023 The MathWorks, Inc.

    % The TensorListReserve operator returns a TensorList tensor:
    %   tf.Tensor(<TensorList>, shape=(), dtype=variant)
    % 
    % Hence, tfTensorListReserve returns an array of structs, each holding a 
    % dlarray and its rank.
    % 
    % element_shape: The only valid scalar shape tensor is the fully unknown 
    % shape specified as -1 can only be -1 if it is a scalar else it has to
    % be a rank 1 tensor containing the shape of the tensor elements to be 
    % reserved, hence this will always be in forward TF format
    elementShapeVal = elementShape.value; 
    elementShapeRank = elementShape.rank;

    % num_elemets: The num_elements to reserve must be a non negative scalar
    numElementsVal = numElements.value;

    y = [];

    if elementShapeRank == 0 && elementShapeVal == -1
        elemTensor = dlarray.empty;
        element = struct('value', elemTensor, 'rank', -1);        
    else
        elemRank = numel(elementShapeVal);
        if elemRank <= 1
            % Forward TF format
            elemTensor = dlarray(zeros([elementShapeVal 1]));
            element = struct('value', elemTensor, 'rank', elemRank);
        else
            % Reverse TF format as rank > 1
            elementShapeVal(elementShapeVal==-1) = [];
            elemTensor = dlarray(zeros(flip(elementShapeVal)));
            element = struct('value', elemTensor, 'rank', elemRank);
        end         
    end
    
    for i = 1:numElementsVal
        y = [y element]; %#ok<AGROW>
    end
end
