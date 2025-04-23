function result = translateSplit(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2021-2023 The MathWorks, Inc.
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    num_splits = node_def.attr.num_split.i;

    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfSplit", ...
                    {MATLABOutputName}, [MATLABArgIdentifierNames, num_splits]);

    result.NumOutputs = str2double(num_splits); 
    result.OpFunctions = "tfSplit";
    result.Success = true; 
end