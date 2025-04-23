function y = tfBroadCastTo(input, shape)

%   Copyright 2020-2024 The MathWorks, Inc.

x = input.value; 

if isstruct(shape)
    shape = shape.value;
end

shape = flip(shape); 
outputRank = numel(shape); 

if isdlarray(x) % in reverse TF format
    x = stripdims(x); 
end 

inShape = size(x); 
expandedShape = ones(1, numel(shape)); 
expandedShape(1:numel(inShape)) = inShape; 
y = x; 
for i = 1:numel(shape) 
    if expandedShape(i) == 1 && shape(i) ~= 1
        rmsize = num2cell(ones(1, numel(shape))); 
        rmsize{i} = shape(i); 
        y = repmat(y, rmsize{:}); 
    end
end

y = struct('value', y, 'rank', outputRank); 
end 
