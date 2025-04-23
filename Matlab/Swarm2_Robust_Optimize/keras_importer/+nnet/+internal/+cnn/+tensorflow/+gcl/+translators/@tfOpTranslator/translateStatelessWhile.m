function result = translateStatelessWhile(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.

%   output = input; While (Cond(output)) { output = Body(output) }

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    input = MATLABArgIdentifierNames(:)';
    
    condFunc = node_def.attr.cond.func; 
    condFuncName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({condFunc.name}); 
    condFuncName = condFuncName{1}; 
    
    bodyFunc = node_def.attr.body.func; 
    bodyFuncName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({bodyFunc.name}); 
    bodyFuncName = bodyFuncName{1}; 
    

    if numOutputs > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
    else
        outputNames = {MATLABOutputName}; 
    end
    
    % Write the caller of this function
    condMap = string(outputNames) + "=" + string(input) + ";" + newline;
    condMap = strjoin(condMap);
    
    condInputs = nnet.internal.cnn.tensorflow.gcl.util.iSplitFcnCalls(string([outputNames {this.LAYERREF}]), numel(condFuncName)); 
    condCall = sprintf("%s(%s)", condFuncName, strjoin(condInputs, ", "));    
    bodyFuncCall = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(bodyFuncName, outputNames, [outputNames {this.LAYERREF}]);    
    
    result.Code = condMap + "while getfield(" + condCall + ",'value')" + newline + bodyFuncCall + newline + "end"  + newline; 
    
    % Log the name of the called function to be generated later. 
    result.SubFunctions = {condFunc.name, bodyFunc.name};
    result.Success = true; 
end 
