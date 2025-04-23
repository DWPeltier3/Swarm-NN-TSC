function [str] = writeFunctionCall(functionName, outputs, inputs)
%WRITEFUNCTIONCALL a subroutine that writes a function call to a string. 
% functionName is a string. outputs and inputs are cell strings.

%   Copyright 2020-2023 The MathWorks, Inc.

    inputs = string(inputs);
    outputs = string(outputs); 
    inputs = nnet.internal.cnn.tensorflow.gcl.util.iSplitFcnCalls(inputs, numel(functionName)); 
    outputs = nnet.internal.cnn.tensorflow.gcl.util.iSplitFcnCalls(outputs);
    if numel(outputs) > 1
        str = sprintf("[%s] = %s(%s);", strjoin(outputs, ", "), functionName, strjoin(inputs, ", "));
    else
        str = sprintf("%s = %s(%s);", strjoin(outputs, ", "), functionName, strjoin(inputs, ", "));
    end
end
