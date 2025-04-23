function result = translateBatchMatMulV2(this, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.
    import nnet.internal.cnn.keras.util.*;
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;

    if (isfield(node_def.attr,'adj_x') && node_def.attr.adj_x.b) || (isfield(node_def.attr,'adj_y') && node_def.attr.adj_y.b)
        this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnsupportedBatchMatMulV2WithAdjoint', MessageArgs={node_def.op});
        result.Code = "error('Adjointed matrix multiplication is not supported for the BatchMatMulV2 operator.');"; 
        result.IsCommenting = true; 
        result.Comment = "% Unsupported operator: " + node_def.op + newline + "% output: " + MATLABOutputName + newline + "% inputs: " + strjoin(MATLABArgIdentifierNames, ","); 
        result.Node = node_def; 
        result.Success = false; 
    else
        result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfBatchMatMulV2", {MATLABOutputName}, MATLABArgIdentifierNames); 
        result.OpFunctions = "tfBatchMatMulV2";
        result.Success = true; 
    end
end 