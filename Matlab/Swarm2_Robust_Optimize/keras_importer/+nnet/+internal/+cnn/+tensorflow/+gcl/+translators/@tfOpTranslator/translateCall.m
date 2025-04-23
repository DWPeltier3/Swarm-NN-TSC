function result = translateCall(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    func = node_def.attr.f.func; 
    callname = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({func.name}); 
    callname = callname{1}; 

    if numOutputs > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
    else
        outputNames = {MATLABOutputName}; 
    end
    
    % Write the caller of this function
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(callname, outputNames, [MATLABArgIdentifierNames {this.LAYERREF}]);
    
    % Log the name of the called function to be generated later. 
    result.SubFunctions = {func.name};
    result.Success = true; 
end 
