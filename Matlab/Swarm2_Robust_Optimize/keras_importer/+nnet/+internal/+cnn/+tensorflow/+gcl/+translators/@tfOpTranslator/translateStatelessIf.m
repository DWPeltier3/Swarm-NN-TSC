function result = translateStatelessIf(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2022-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    cond = MATLABArgIdentifierNames{1};
    input = MATLABArgIdentifierNames(2:end);
    
    thenFunc = node_def.attr.then_branch.func; 
    thenFuncName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({thenFunc.name}); 
    thenFuncName = thenFuncName{1}; 
    
    elseFunc = node_def.attr.else_branch.func; 
    elseFuncName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({elseFunc.name}); 
    elseFuncName = elseFuncName{1};

    if numOutputs > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
    else
        outputNames = {MATLABOutputName}; 
    end
    
    % Write the caller of this function
    ifCondCall = ['tfStatelessIf(' cond ')'];
    thenFuncCall = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(thenFuncName, outputNames, [input {this.LAYERREF}]);
    elseFuncCall = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(elseFuncName, outputNames, [input {this.LAYERREF}]);
    
    
    result.Code = "if " + ifCondCall + newline + thenFuncCall + newline + "else" + newline + elseFuncCall + newline + "end"  + newline; 
    
    % Log the name of the called function to be generated later. 
    result.SubFunctions = {thenFunc.name, elseFunc.name};
    result.OpFunctions = "tfStatelessIf"; 
    result.Success = true; 
end 
