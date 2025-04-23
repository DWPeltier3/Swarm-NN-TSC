function result = translateConst(this, node_def, MATLABOutputName, TFConstants)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    
    result.Code = MATLABOutputName + " = " + writeConstant(this, node_def, MATLABOutputName, TFConstants); 
    result.NumOutputs = 1;    
    result.Success = true; 
end
