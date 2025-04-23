function inputOrOutput = iSplitFcnCalls(inputOrOutput, initialLen)
    % This is a pre-processing step before a call to writeFunctionCall or
    % writeMATLABSignature. 
    % inputOrOutput is a cell array of strings which represent a function
    % input or output 
    % initialLen is an optional argument to specify the start position of
    % the arguments. Defaults to zero 

%   Copyright 2020-2021 The MathWorks, Inc.

    if nargin < 2
        initialLen = 0; 
    end 
    maxlinelen = 100; 
    curlinelen = initialLen; 
    for i = 1:numel(inputOrOutput)
        curlinelen = curlinelen + numel(inputOrOutput{i}); 
        if curlinelen > maxlinelen
            if i ~= numel(inputOrOutput)
                inputOrOutput{i + 1} = ['...' newline inputOrOutput{i + 1}]; 
            end 
            curlinelen = 0; 
        end 
    end
end 
