function y = tfGather(params, indices, axis, batchDims)

%   Copyright 2020-2024 The MathWorks, Inc.
    
    paramsVal = params.value; 
    paramsRank = params.rank;
    indicesVal = indices.value;
    indicesRank = indices.rank;
    
    if nargin < 4
        batchDims = 0;
    end

    if paramsRank == 1
        paramsVal = paramsVal(:);
    end

    % Input is in reverse TF format
    % Convert it to fwd TF format
    if paramsRank > 1
        paramsVal = permute(paramsVal, paramsRank:-1:1); 
    end
    
    % Indices is in reverse TF format
    % Convert it to fwd TF format
    if indicesRank > 1
        indicesVal = permute(indicesVal, indicesRank:-1:1); 
    end 

    xShape =  getTensorShape(paramsVal, paramsRank);
    indicesShape = getTensorShape(indicesVal, indicesRank);

    if isstruct(axis)
        axisVal = axis.value;
    end

    % Handle negative axis values
    if axisVal < 0
       axisVal = paramsRank + axisVal;
    end

    % Now xVal and indicesVal should be in forward TF format
    mlAxis = axisVal + 1;
    mlInd = indicesVal + 1; 
    
    ind.type = '()';
    
    if batchDims > 0 && batchDims == indicesRank
        % need to get full subscripts for indexing
        [subsInds] = getFullSubsIndices(xShape, indicesShape, mlInd);
        y = [];
        [r,~] = size(subsInds);
        for j = 1:r
            ind.subs = num2cell(subsInds(j,:));
            y(end+1) = subsref(paramsVal, ind); %#ok<AGROW>
        end
        yShape = [xShape(1:mlAxis-1) indicesShape(batchDims + 1:end) xShape(mlAxis +1: end)];
        if numel(yShape) == 1
            % MATLAB reshape requires at-least 2 elements for reshape vector
            yShape = [yShape 1];
        end
        yRank = numel(yShape);
        y = reshape(y,flip(yShape));
        y = permute(y, yRank:-1:1);
    elseif batchDims > 0 && axisVal == batchDims && batchDims <= indicesRank
        [subsInds] = getPartialSubsIndices(indicesShape, indicesRank, mlInd, batchDims);
        xShapeAfterIdx = xShape(indicesRank+1:end);
        y = [];
        [r,~] = size(subsInds);
        for j = 1:r
            c = num2cell(subsInds(j,:));
            c(end+1:end+(paramsRank-indicesRank)) = {':'}; 
            numelX = prod(xShapeAfterIdx);
            y(end+1:end+numelX) = paramsVal(c{:}); 
        end        
        yShape = [xShape(1:mlAxis-1) indicesShape(batchDims + 1:end) xShape(mlAxis +1: end)];
        if numel(yShape) == 1
            % MATLAB reshape requires at-least 2 elements for reshape vector
            yShape = [yShape 1];
        end
        yRank = numel(yShape);
        y = reshape(y,flip(yShape));
        y = permute(y, yRank:-1:1);
    else
        ind.subs = repmat({':'}, 1, paramsRank);
        ind.subs{mlAxis} = mlInd(:);
        y = subsref(paramsVal, ind);
        if batchDims > 0
            yShape = [xShape(1:mlAxis-1) indicesShape(batchDims + 1:end) xShape(mlAxis +1: end)];
            yRank = numel(yShape); 
        else
            yRank = paramsRank - 1 + indicesRank;
            yShape = [xShape(1:mlAxis-1) indicesShape xShape(mlAxis +1: end)];
        end
        if numel(yShape) == 1
            % MATLAB reshape requires at-least 2 elements for reshape vector
            yShape = [yShape 1];
        end               
        y = reshape(y, yShape);
    end   
    
    if yRank > 1
        y = permute(y, yRank:-1:1);
    end
    y = struct('value', y, 'rank', yRank);

end


function [subsInds] = getPartialSubsIndices(indShape, indRank, mlInd, batchDims)
    mlInd = permute(mlInd, indRank:-1:1);
    partialSubsInds = mlInd(:); % flattened subscript indices
    batchDimsArr = {};    
    for i = 1: numel(indShape)
        batchDimsArr(i) = {1:indShape(i)}; %#ok<AGROW>
    end    
    batchSubInds = table2array(combinations(batchDimsArr{:})); 
    subsInds = horzcat(batchSubInds(:,1:batchDims),partialSubsInds);
end 

function [subsInds] = getFullSubsIndices(xShape, indShape, mlInd)
    partialSubsInds = mlInd(:); % flattened subscript indices
    numSpecsInBatchDims = prod(indShape);
    batchDimsArr = {};    
    for i = 1: numel(indShape)
        batchDimsArr(i) = {1:indShape(i)}; %#ok<AGROW>
    end    
    batchSubInds = table2array(combinations(batchDimsArr{:})); 
    subsInds = horzcat(batchSubInds,partialSubsInds);
    
    extraDimsInXShape = xShape(numel(indShape)+2 : end);
    numSpecsInExtraDimsInXShape = prod(extraDimsInXShape);
    extraNonBatchDimsArr = {};    
    for j = 1: numel(extraDimsInXShape)
        extraNonBatchDimsArr(j) = {1:extraDimsInXShape(j)}; %#ok<AGROW>
    end    
    extraNonBatchDimsArr = table2array(combinations(extraNonBatchDimsArr{:}));
    extraNonBatchDimsArr = repmat(extraNonBatchDimsArr, [numSpecsInBatchDims 1]);
    subsInds = repelem(subsInds, numSpecsInExtraDimsInXShape, 1);    
    subsInds = horzcat(subsInds, extraNonBatchDimsArr);     
end 

function [shape] = getTensorShape(xVal, xRank)
    % Assumes that xval is in fwd-TF format
    shape = size(xVal);
    nShape = numel(shape);
    if nShape < xRank
        % Add trailing singleton dims
        shape(end+1:end+xRank-nShape) = 1;
    elseif nShape > xRank
        shape = shape(1:xRank);
    end
end 