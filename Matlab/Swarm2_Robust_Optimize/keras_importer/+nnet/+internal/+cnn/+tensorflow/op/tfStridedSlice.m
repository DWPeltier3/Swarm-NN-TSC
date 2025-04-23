function y = tfStridedSlice(x, beginIdx, endIdx, strideIdx, beginMask, endMask, ellipsisMask, newAxisMask, shrinkMask)
%{{import_statement}}

%   Copyright 2020-2023 The MathWorks, Inc.

xVal = x.value;
xRank = x.rank;

if isempty(xVal)
    % Nothing to slice
    y = x;
    return;
end

beginIdxVal = beginIdx.value;
endIdxVal = endIdx.value;
strideIdxVal = strideIdx.value;

nSpecs = numel(beginIdxVal); 
if isempty(beginMask)
    beginMask = 0; 
end
if isempty(endMask)
    endMask = 0;
end
if isempty(ellipsisMask)
    ellipsisMask = 0; 
end
if isempty(newAxisMask)
    newAxisMask = 0; 
end
if isempty(shrinkMask)
    shrinkMask = 0; 
end

beginMask = fliplr(dec2bin(beginMask, nSpecs)); 
endMask = fliplr(dec2bin(endMask, nSpecs));
ellipsisMask = fliplr(dec2bin(ellipsisMask, nSpecs)); 
newAxisMask = fliplr(dec2bin(newAxisMask, nSpecs)); 
shrinkMask = fliplr(dec2bin(shrinkMask, nSpecs)); 

if xRank > 1
    xVal = permute(xVal, xRank:-1:1);
end

% xVal should be in fwd TF order now
xShape = size(xVal);
if (numel(xShape)) < xRank
    % Add back dropped singletons
    numDroppedDims = xRank - numel(xShape);
    xShape(end+1:end+numDroppedDims) = 1;
elseif (numel(xShape)) > xRank
    % only in case of rank 1
    xShape(end) = [];
end

newShape = xShape;
for i = 1:nSpecs
    if (newAxisMask(i)=='1')
        % New axis added at the i'th dimension
        newShape(i+1:end+1) = newShape(i:end);
        newShape(i) = 1;
    elseif (ellipsisMask(i)=='1')
        if (numel(newShape) - i) > (nSpecs - i)
            % More than one axis covered by elipsis starting at i
            % Expand the Specs            
            nDimsElipsis = (numel(newShape) - i) - (nSpecs - i);
            nSpecs = nSpecs + nDimsElipsis;
            beginIdxVal(i+nDimsElipsis+1:end+nDimsElipsis) = beginIdxVal(i+1:end);
            endIdxVal(i+nDimsElipsis+1:end+nDimsElipsis) = endIdxVal(i+1:end);
            strideIdxVal(i+nDimsElipsis+1:end+nDimsElipsis) = strideIdxVal(i+1:end);
            beginMask(i+nDimsElipsis+1:end+nDimsElipsis) = beginMask(i+1:end);
            endMask(i+nDimsElipsis+1:end+nDimsElipsis) = endMask(i+1:end);
            ellipsisMask(i+nDimsElipsis+1:end+nDimsElipsis) = ellipsisMask(i+1:end);
            newAxisMask(i+nDimsElipsis+1:end+nDimsElipsis) = newAxisMask(i+1:end);
            shrinkMask(i+nDimsElipsis+1:end+nDimsElipsis) = shrinkMask(i+1:end);          
            
            % Set the masks to default values for the expanded specs
            beginIdxVal(i+1:i+nDimsElipsis) = 0;
            endIdxVal(i+1:i+nDimsElipsis) = 0;
            strideIdxVal(i+1:i+nDimsElipsis) = 1;        
            beginMask(i+1:i+nDimsElipsis) = '1';
            endMask(i+1:i+nDimsElipsis) = '1';
            ellipsisMask(i+1:i+nDimsElipsis) = '0';
            newAxisMask(i+1:i+nDimsElipsis) = '0';
            shrinkMask(i+1:i+nDimsElipsis) = '0';
        end        
    end
end

if xRank > nSpecs && ~any(find(ellipsisMask=='1'))
    % If the indexing dimensions are less than the rank of the input tensor
    % and ellipsisMask is all 0's
    % pad beginIdx, endIdx and strideIdx with default values
    numUnSpecDims = xRank - nSpecs;
    % get fwd TF shape to fill the remaining endIdx
    endIdxVal(end+1 : end+numUnSpecDims) = xShape(nSpecs+1:end);
    beginIdxVal(end+1 : end+numUnSpecDims) = 0;
    strideIdxVal(end+1 : end+numUnSpecDims) = 1;
    
    % pad the masks with 0's
    beginMask(end+1 : end+numUnSpecDims) = '0';
    endMask(end+1 : end+numUnSpecDims) = '0';
    shrinkMask(end+1 : end+numUnSpecDims) = '0';
    ellipsisMask(end+1 : end+numUnSpecDims) = '0';
    newAxisMask(end+1 : end+numUnSpecDims) = '0';
    nSpecs = nSpecs + numUnSpecDims;
end

yRank = numel(newShape);
if numel(newShape) > 1
    xVal = reshape(xVal, newShape);     
end				  

S = substruct('()', {});
for i = 1:nSpecs
    % Build the substruct for each dimension.
    if (newAxisMask(i)=='1')
        S.subs{i} = 1:size(xVal, i);
    elseif (ellipsisMask(i)=='1')
        % Ellipsis used for indexing the i'th dimension
        % select all elements from that dimension
        S.subs{i} = 1:size(xVal, i);
    elseif (shrinkMask(i)=='1')
        S.subs{i} = beginIdxVal(i) + 1;        
    else
        curBeginIdx  = mod(beginIdxVal(i), size(xVal,i)) + 1; 
        curStrideIdx = strideIdxVal(i);

        if strcmp(beginMask(i), '1')
            if curStrideIdx >= 0      
                curBeginIdx = 1; 
            else
                curBeginIdx = size(xVal, i); 
            end
        end 

        if strcmp(endMask(i), '1')
            if curStrideIdx >= 0      
                curEndIdx = size(xVal, i);
            else
                curEndIdx = 0; 
            end
        else
            if endIdxVal(i) >= 0
                if curStrideIdx >= 0      
                    curEndIdx = endIdxVal(i);
                else
                    curEndIdx = endIdxVal(i) + 1;
                end                 
            else 
                if curStrideIdx >= 0      
                    curEndIdx = mod(endIdxVal(i), size(xVal, i));
                else
                    curEndIdx = mod(endIdxVal(i), size(xVal, i)) + 1;
                end
            end
        end
    
        if curStrideIdx >= 0        
            idxVec = curBeginIdx:curStrideIdx:curEndIdx; 
        else
            if curEndIdx > curBeginIdx
                idxVec = curEndIdx:curStrideIdx:curBeginIdx; 
            else
                curEndIdx = curEndIdx + 1;
                idxVec = curBeginIdx:curStrideIdx:curEndIdx ;
            end
        end         
        S.subs{i} = idxVec;         
    end
end 

% Return the indexed array x, with the same number of dimensions.
yVal = subsref(matlab.lang.internal.move(xVal), S); 

if yRank > 1
    % permute back to reverse TF
    yVal = permute(yVal, yRank:-1:1);
end
  
y = struct('value', yVal, 'rank', yRank); 

if(any(find(shrinkMask=='1')))
    % tfSqueeze expects 0-based axes (from TF)
    y = tfSqueeze(y, find(shrinkMask=='1') - 1); 
end
end

