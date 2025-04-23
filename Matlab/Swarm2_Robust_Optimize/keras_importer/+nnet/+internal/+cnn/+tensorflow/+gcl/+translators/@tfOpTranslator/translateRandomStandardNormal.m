function result = translateRandomStandardNormal(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    dtype = 'single'; 
    if isfield(node_def.attr, 'dtype')
        dtype = "'" + node_def.attr.dtype.type + "'";
    end
    
    if isfield(node_def.attr, 'seed') || isfield(node_def.attr, 'seed2')
        this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnsupportedRandStdNormalOpWithSeedOrSeed2', MessageArgs={node_def.op});        
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfRandomStandardNormal", {MATLABOutputName}, [MATLABArgIdentifierNames {dtype}]);
    result.OpFunctions = "tfRandomStandardNormal";
    result.Success = true;
end
