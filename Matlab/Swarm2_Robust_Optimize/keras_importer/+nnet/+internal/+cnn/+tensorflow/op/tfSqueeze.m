function y = tfSqueeze(input, axis)

%   Copyright 2020-2023 The MathWorks, Inc.
    
    inputVal = input.value;
    inputRank = input.rank;
    
    % If axis is a struct extract the numeric axis value
    if isstruct(axis)
        axisVal = axis.value;
        if isdlarray(axisVal)
            axisVal = extractdata(axisVal);
        end
    else
        % if input is numeric
        axisVal = axis;
    end
        
    reshapeVector = [];
    outputRank = inputRank; 
    if isempty(axisVal)
        % No axis specified, remove all 1 dimensions
        for i = 1:inputRank
            if size(inputVal, i)~=1            
                reshapeVector(end+1) = size(inputVal, i); %#ok<AGROW>
            else
                outputRank = outputRank - 1; 
            end
        end
    else
        % Axis specified, only remove dimensions of size 1 at the specified
        % axes values
        if any(axisVal < 0)
        % Handle negative axis values
            negIdx = axisVal < 0;
            axisVal(negIdx) = mod(axisVal(negIdx), inputRank);
        end        
        MLAxis = inputRank - axisVal;
        
        for i = 1:inputRank
            if ismember(i, MLAxis) && size(inputVal,i) == 1
                outputRank = outputRank - 1;                
            else
                reshapeVector(end+1) = size(inputVal, i); %#ok<AGROW>
            end
        end
    end
    
    % reshapeVector has dimension sizes in reverse TF format
    % x is in reverse TF format
    if numel(reshapeVector) > 1
        yVal = reshape(matlab.lang.internal.move(inputVal), reshapeVector);
    else
        yVal = inputVal(:);
    end

    y = struct('value', yVal, 'rank', outputRank);
end
