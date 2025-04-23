function z = tfTensorListSetItem(inputHandle, index, item)

%   Copyright 2023 The MathWorks, Inc.

    % The TensorListSetItem operator sets the index-th position of the input list (input_handle) to contain the given tensor:
    %   tf.Tensor(<TensorList>, shape=(), dtype=variant)
    % 
    % Hence, tfTensorListSetItem returns returns an array of structs, each holding a 
    % dlarray and its rank.    

    indexVal = index.value + 1;
    itemVal = item.value;
    itemRank = item.rank;

    z = inputHandle;
    z(indexVal).value = itemVal;
    z(indexVal).rank = itemRank;
end






