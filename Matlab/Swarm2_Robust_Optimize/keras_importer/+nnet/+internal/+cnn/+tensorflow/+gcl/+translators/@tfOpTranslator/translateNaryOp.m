function result = translateNaryOp(~, tfOpFcnName, MATLABOutputName, MATLABArgIdentifierNames, tfOpDependencies)
    % Translate a node into code that calls hand-written tfOp functions that 
    % expects N inputs. 

%   Copyright 2020-2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(tfOpFcnName, {MATLABOutputName}, MATLABArgIdentifierNames);
    
    result.NumOutputs = 1;
    if nargin < 5
        result.OpFunctions = tfOpFcnName;
    else
        tfOpDependencies(end+1) = tfOpFcnName;
        result.OpFunctions = tfOpDependencies;
    end
    result.Success = true; 
end 
