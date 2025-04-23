function y = tfAll(input, axis, keepDims)

%   Copyright 2022-2023 The MathWorks, Inc.
   
    inputVal = input.value;
    inputRank = input.rank;
    yRank = inputRank;
    axisVal = axis.value;

    if inputRank == 0
        y = struct('value', inputVal, 'rank', inputRank);
        return
    end

    if any(axisVal < 0)
    % Handle negative axis values
        negIdx = axisVal < 0;
        axisVal(negIdx) = inputRank + axisVal(negIdx);
    end
       
    % reverse TF
    mlAxis = inputRank - axisVal;
    y = all(inputVal, mlAxis);

   if nargin < 3
       keepDims = false;
   end

   if ~keepDims
        outSize = ones(1, inputRank); 
        outSize(1:ndims(y)) = size(y); 
        outSize(mlAxis) = [];
        yRank = inputRank - numel(mlAxis); 
        if yRank > 1
            y = reshape(y, outSize);
        else 
            reshape(y, [outSize 1 1]); 
        end 
   end
   
    y = dlarray(single(y));
    y = struct('value', y, 'rank', yRank);
end
