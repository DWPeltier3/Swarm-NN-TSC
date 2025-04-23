function result = translateAssert(~, MATLABArgIdentifierNames)
%

%   Copyright 2022-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = "assert(logical(" + MATLABArgIdentifierNames{1} + ".value));";
    result.Success = true; 
end 
