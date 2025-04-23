function outputNames = makeMultipleOutputArgs(~, outputName, numOutputs)
    % Returns a cell array of outputName with cell array bracing.
    % This is a utility for writing multiple output arguments.

%   Copyright 2020-2021 The MathWorks, Inc.

    indices = 1:numOutputs;
    outputNames = cellstr(outputName + "{" + indices + "}");
end
