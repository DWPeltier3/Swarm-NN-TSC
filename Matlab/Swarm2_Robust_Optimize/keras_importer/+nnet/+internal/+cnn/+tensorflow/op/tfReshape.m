function y = tfReshape(tensor, shape)

%   Copyright 2020-2024 The MathWorks, Inc.

    newShape = shape.value;
    tensorVal = tensor.value;
    outRank = numel(newShape);     
    newShape = num2cell(newShape(:)'); 
    for i = 1:outRank
        if newShape{i} == -1
            newShape{i} = []; 
        end 
    end 

    if outRank == 1
        % for a rank-1 tensor output append a 1 to the new shape
        newShape{end + 1} = 1;
    end
    
    % Input is already in reverse TensorFlow ordering.
    % A row major reshape will require us to flip the reshape values.        
    if outRank > 1
        yVal = reshape(tensorVal, newShape{end:-1:1});
    else
        yVal = reshape(tensorVal, newShape{1:end}); 
    end
    
    y = struct('value', yVal, 'rank', outRank);
end
