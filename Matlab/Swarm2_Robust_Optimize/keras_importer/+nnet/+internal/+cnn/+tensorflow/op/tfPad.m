function y = tfPad(input, paddings, constantValues)

%   Copyright 2020-2023 The MathWorks, Inc.

    inputRank = input.rank;
    inputVal = input.value;

    if ~isa(inputVal, 'dlarray')
        % If a numeric array permute the input tensor to match Forward TF Format.   
        inputVal = permute(inputVal, inputRank:-1:1);    
    elseif inputRank > 1
        % Assume the input dlarray to be in reverse TF order and permute to
        % forward TF
        inputVal = permute(inputVal, inputRank:-1:1);
    end       

    % paddings is a Nx2 matrix. Where N is the number of dimensions of x. 
    % For a dimension D paddings(D, 1) is the number of values to pad
    % before the contents of x in that dimension. and paddings(D, 2) is the
    % number of values to add after the contents of x in that dimension. 
    % We always expect paddings to be a numeric array or an unformatted dlarray
    % It will always be in reverse TF format.
    paddings = paddings.value';
    
    if isstruct(constantValues)
        constantValues = constantValues.value; 
    end
    
    sizeInput = size(inputVal); 
    if inputRank == 1
        % rank 1 input, its a vector 
        sizeInput = size(inputVal, 1);
    elseif inputRank > numel(sizeInput)
        % add back potentially dropped dims 
        diff = inputRank - numel(sizeInput); 
        sizeInput(end+1:end+diff) = 1; 
    end 

    sizeY = zeros(1, inputRank); 
    for i = 1:size(paddings, 1)
        sizeY(i) = size(inputVal, i) + paddings(i, 1) + paddings(i, 2); 
    end 
    
    if inputRank == 1
        % Padding does not change the rank of the input
        sizeY = [sizeY 1];
    end

    % Construct output array with padded size. 
    yVal = dlarray(cast(constantValues, 'like', inputVal) * ones(sizeY, 'like', inputVal)); 

    % Construct subsref indices for inserting (and cropping) the original
    ySubs = cell(1, size(paddings, 1)); 
    inputSubs = cell(1, size(paddings, 1)); 

    for i=1:numel(sizeInput)
        ySubs{i} = max(1,1+paddings(i,1)) : min(sizeY(i), sizeY(i)-paddings(i, 2));
        inputSubs{i} = max(1,1-paddings(i,1)) : min(sizeInput(i), sizeInput(i)+paddings(i, 2));
    end

    sY      = struct('type', '()');
    sY.subs = ySubs;
    sInput      = struct('type', '()');
    sInput.subs = inputSubs;

    % Insert/crop the original into the result
    yVal = subsasgn(yVal, sY, subsref(inputVal, sInput));
    
    % Permute to reverse TF ordering. 
    if inputRank > 1
        yVal = permute(yVal, inputRank:-1:1);
    end
    
    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', inputRank);
end
