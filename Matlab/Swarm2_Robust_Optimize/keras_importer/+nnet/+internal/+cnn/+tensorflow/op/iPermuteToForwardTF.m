function fwdTFInput = iPermuteToForwardTF(revTFInput, rank)
% Permutes the data from reverse to forward TensorFlow format order

%   Copyright 2023-2024 The MathWorks, Inc.

    if isdlarray(revTFInput) && rank >= 2    
        inputLabels = revTFInput.dims;
        % Only perform permute if the input data is all U-labeled or
        % unformatted
        if all(inputLabels == 'U') || isempty(inputLabels)
            fwdTFInput = dlarray(permute(revTFInput, rank:-1:1), repmat('U', [1 rank]));
        else 
            fwdTFInput = revTFInput;
        end
    else
        fwdTFInput = revTFInput;
    end

  
end