function [stacked] = tfPack(axis, varargin)

%   Copyright 2022-2023 The MathWorks, Inc.

nTensors = nargin - 1; 
inRank = varargin{1}.rank;

for i = 1:nTensors
    tempRank = varargin{i}.rank;
    if tempRank > 1
        varargin{i}.value = permute(varargin{i}.value, tempRank:-1:1);
    end
end

% Account for dropped trailing singleton dimensions
inputShape = ones(1, inRank);
inVal = varargin{1}.value;
inputShape(1:ndims(inVal)) = size(inVal);

% varargin should be in forward TF format now.  
if isempty(axis)
    axis = 0;
end

if axis < 0
    % handle negative axis values
    if axis == (-1 * (inRank+1))
        mlAxis = -1;
    else
        mlAxis = mod(axis, inRank);
    end
else 
    mlAxis = axis; 
end

% inputShape holds the shape in forward TF format
if (inRank == 0)
    % if the input is a scalar
    outputShape = [1 1];
else
    % If the input rank > 0, stack on the dimension specified by axis
    if axis < 0
        outputShape = [inputShape(1:mlAxis + 1) 1 inputShape(mlAxis + 2:end)];
    else
        outputShape = [inputShape(1:mlAxis) 1 inputShape(mlAxis + 1:end)];
    end
end

for i = 1:nTensors 
    varargin{i} = reshape(varargin{i}.value, outputShape);   
end

outRank = inRank + 1;

if axis < 0
    stackedVal = cat(mlAxis + 2, varargin{:}); 
else
    stackedVal = cat(mlAxis + 1, varargin{:}); 
end

if outRank > 1 
    % convert to reverse TF format if rank > 1
    stackedVal = permute(stackedVal, outRank:-1:1);
end

stackedVal = dlarray(stackedVal); 
stacked = struct('value', stackedVal, 'rank', outRank); 
end
