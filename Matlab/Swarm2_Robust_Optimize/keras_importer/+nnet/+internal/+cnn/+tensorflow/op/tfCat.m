function y = tfCat(axis, varargin)

%   Copyright 2020-2023 The MathWorks, Inc.
    
    % All input tensors should have the same rank.
    allRanks = cellfun(@(x)x.rank, varargin);
    if numel(unique(allRanks)) ~= 1
        error('tfCat: Ranks of all input tensors should match for ConcatV2.');
    end

    outputRank = allRanks(1);

    % If axis is a struct extract the numeric axis value
    if isstruct(axis)
        axisVal = axis.value;
    else
        % if input is numeric
        axisVal = axis;
    end
    
    % Handle negative axis value
    if axisVal < 0
        mlAxis = axisVal - floor(axisVal./outputRank).*outputRank;
    else 
        mlAxis = axisVal; 
    end

    mlAxis = outputRank - mlAxis;

    isDlarray = cellfun(@(x)isdlarray(x.value), varargin); 
    if any(isDlarray) 
        % if any inputs are dlarrays, all values must be cast to the same
        % type and converted to TensorFlow dimension format. 
        for i = 2:numel(varargin)
            varargin{i}.value = cast(varargin{i}.value, 'like', varargin{1}.value);
        end
    end

    nonEmptyTensors = {};
    j = 1;
    for i = 1:numel(varargin)
        varargin{i} = varargin{i}.value;
        % remove any empty tensors
        if ~isempty(varargin{i})
            nonEmptyTensors{j} = varargin{i}; %#ok<AGROW>
            j = j + 1;
        end
    end
        
    % concatenate all inputs (in reverse TF format)
    % as long as one of the inputs has labels the output of 'cat' will have labels.
    yVal = cat(mlAxis, nonEmptyTensors{:}); 
    
    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', outputRank);
    
end 
