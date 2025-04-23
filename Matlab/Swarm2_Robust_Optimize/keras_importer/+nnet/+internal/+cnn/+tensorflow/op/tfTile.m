function y = tfTile(input, multiples)
   
    % Copyright 2020-2024 The MathWorks, Inc.

    inputVal = input.value; 
    inputRank = input.rank; 
    
    if isstruct(multiples)
        multiplesVal = multiples.value;
    else
        multiplesVal = multiples;
    end

    % If inputVal is a dlarray extract the numeric value
    if isdlarray(inputVal)
        inputVal = stripdims(inputVal);
    end
    
    % xval is in reverse TF format
    if inputRank > 1
        inputVal = permute(inputVal, inputRank:-1:1); 
    end
    % Now xval is in Forward TF format

    for i = 1:numel(multiplesVal) 
        repVec = ones(1, numel(multiplesVal)); 
        repVec(i) = multiplesVal(i); 
        inputVal = repmat(inputVal, repVec); 
    end 
    
    % permute xval back to reverse TF format
    if inputRank > 1
        yVal = permute(inputVal, inputRank:-1:1); 
    end

    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', inputRank); 
end
