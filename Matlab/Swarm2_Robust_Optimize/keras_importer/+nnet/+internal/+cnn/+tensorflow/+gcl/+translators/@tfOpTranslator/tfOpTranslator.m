classdef tfOpTranslator 
    % tfOpTranslator: This class is used to create NodeTranslationResults
    % from original TensorFlow nodes. 

%   Copyright 2020-2023 The MathWorks, Inc.
    
    properties (Access = protected)
        LAYERREF = "obj" % A string storing the name of the layer where constants are stored 
        RANKFIELDNAME = "rank" % A string storing the name of the structure field storing tensor rank 
    end

    properties (Access = public)
        ImportManager % ImportManager object reference used for logging translation warnings
        LayerName % Holds the name of the generated custom layer for which this translator was called
    end
    
    methods (Access = public)
        function result = translateTFOp(this, node_def, nameMap, constants, numOutputMap, multiOutputNameMap, tfModelNodeToModelOut)
            % Prepare new identifiers and translation 
			import nnet.internal.cnn.keras.util.*;									  
            [MATLABOutputName, MATLABArgIdentifierNames, numOutputs] = prepareNode(this, node_def, nameMap, numOutputMap, multiOutputNameMap, tfModelNodeToModelOut); 
            
            switch node_def.op
                case 'Abs'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "abs", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Add'
                    result = translateBinaryOp(this, "tfAdd", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'AddN'
                    result = translateNaryOp(this, "tfAddN", MATLABOutputName, MATLABArgIdentifierNames, "tfAdd");
                case 'AddV2'
                    result = translateBinaryOp(this, "tfAdd", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'All'
                    result = translateAll(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'ArgMin'
                    result = translateArgMin(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'AssignSubVariableOp'
                    result = translateAssignSubVariableOp(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'AssignVariableOp'
                    result = translateAssignVariableOp(this); 
                case 'AvgPool'
                    result = translateAvgPool(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Assert'
                    result = translateAssert(this, MATLABArgIdentifierNames);  
                case 'BatchMatMulV2'
                    result = translateBatchMatMulV2(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'BiasAdd'
                    result = translateBiasAdd(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                    
                case 'BroadcastTo'
                    result = translateBinaryOp(this, "tfBroadCastTo", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Cast' 
                    result = translateCast(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'ConcatV2'
                    result = translateConcatV2(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'ClipByValue'
                    result = translateNaryOp(this, "tfClipByValue", MATLABOutputName, MATLABArgIdentifierNames);                     
                case 'Const'
                    result = translateConst(this, node_def, MATLABOutputName, constants);                     
                case 'Conv2D'
                    result = translateConv2D(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                    
                case 'DepthToSpace'
                    result = translateDepthToSpace(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);    
                case 'DepthwiseConv2dNative'
                    result = translateDepthwiseConv2dNative(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Equal'
                    isLogicalOp = true;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "eq", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Exp'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "exp", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'ExpandDims'
                    result = translateExpandDims(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Fill'
                    result = translateFill(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'FloorDiv'
                    result = translateBinaryOp(this, "tfFloorDiv", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'FloorMod'
                    result = translateBinaryOp(this, "tfFloorMod", MATLABOutputName, MATLABArgIdentifierNames);
                case 'FusedBatchNormV3'
                    result = translateFusedBatchNorm(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap); 
                case 'GatherV2'
                    result = translateGather(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                     
                case 'Greater'
                    isLogicalOp = true;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "gt", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'GreaterEqual' 
                    isLogicalOp = true;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "ge", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames);  
                case 'Identity'
                    result = translateIdentity(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);  
                case 'IdentityN'
                    result = translateIdentityN(this, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                case 'If'
                    result = translateStatelessIf(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                case 'L2Loss'
                    result = translateUnaryOp(this, "tfL2Loss", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'LeakyRelu'
                    result = translateLeakyRelu(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                     
                case 'Less'
                    isLogicalOp = true;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "lt", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'LessEqual' 
                    isLogicalOp = true;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "le", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Log'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "log", MATLABOutputName, MATLABArgIdentifierNames);
                case 'LogicalAnd' 
                    isLogicalOp = true;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "and", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'MatMul' 
                    result = translateMatMul(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Max'
                    result = translateMax(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);  
                case 'MaxPool'
                    result = translateMaxPool(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                    
                case 'Maximum'
                    result = translateBinaryOp(this, "tfMaximum", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Mean'
                    result = translateMean(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Min'
                    result = translateMin(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);  
                case 'Minimum'
                    result = translateBinaryOp(this, "tfMinimum", MATLABOutputName, MATLABArgIdentifierNames);
                case 'MirrorPad'
                    result = translateMirrorPad(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                     
                case 'Mul'
                    result = translateBinaryOp(this, "tfMul", MATLABOutputName, MATLABArgIdentifierNames);                     
                case 'Neg'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "uminus", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'NoOp'
                    result = translateNoOp(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'NonMaxSuppressionV5'
                    if ~nnet.internal.cnn.keras.isInstalledCVST
                        this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:RequiredProductMissingForOp', MessageArgs={node_def.op, 'Computer Vision Toolbox'});
                    end
                    result = translateNonMaxSuppressionV5(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap); 
                case 'Pack'
                    result = translatePack(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Pad'
                    result = translatePad(this, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'PadV2'
                    result = translatePadV2(this, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'PartitionedCall'
                    result = translateCall(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Pow'
                    isLogicalOp = false;
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedBinaryOp(this, outputDims, "power", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames);                     
                case 'Prod'
                    result = translateProd(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);  
                case 'RandomStandardNormal'
                    result = translateRandomStandardNormal(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Range'
                    result = translateUnaryOp(this, "tfRange", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'ReadVariableOp'
                    result = translateReadVariableOp(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'RealDiv' 
                    result = translateBinaryOp(this, "tfDiv", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Relu'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "relu", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Relu6' 
                    result = translateRelu6(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Reshape' 
                    result = translateNaryOp(this, "tfReshape", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'ResizeBilinear'
                    if ~nnet.internal.cnn.keras.isInstalledIPT
                        this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:RequiredProductMissingForOp', MessageArgs={node_def.op, 'Image Processing Toolbox'});
                    end
                    result = translateResizeBilinear(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'ResizeNearestNeighbor'
                    if ~nnet.internal.cnn.keras.isInstalledIPT
                        this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:RequiredProductMissingForOp', MessageArgs={node_def.op, 'Image Processing Toolbox'});
                    end
                    result = translateResizeNearestNeighbor(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Round'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "round", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Rsqrt'
                    result = translateRsqrt(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'ScatterNd'
                    result = translateNaryOp(this, "tfScatterNd", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Shape'
                    result = translateUnaryOp(this, "tfShape", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Select'
                    result = translateSelect(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Sigmoid'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "sigmoid", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Size'
                    result = translateUnaryOp(this, "tfSize", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Slice'
                    result = translateSlice(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Softmax'
                    result = translateUnaryOp(this, "tfSoftmax", MATLABOutputName, MATLABArgIdentifierNames);
                case 'SpaceToDepth'
                    result = translateSpaceToDepth(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Sqrt'
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "sqrt", MATLABOutputName, MATLABArgIdentifierNames);  
                case 'SquaredDifference' 
                    result = translateSquaredDifference(this, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Split'
                    result = translateSplit(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Square'
                    result = translateSquare(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Squeeze'
                    result = translateSqueeze(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'StatefulPartitionedCall'
                    result = translateCall(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                case 'StatelessIf'
                    result = translateStatelessIf(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                case 'StatelessWhile'
                    result = translateStatelessWhile(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                case 'StopGradient'
                    result = translateUnaryOp(this, "tfStopGradient", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'StridedSlice'
                    result = translateStridedSlice(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Sub' 
                    result = translateBinaryOp(this, "tfSub", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Sum'
                    result = translateSum(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
                case 'Tanh' 
                    outputDims = numel(node_def.attr.x_output_shapes.list.shape.dim);
                    result = translateInlinedUnaryOp(this, outputDims, "tanh", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'TensorListFromTensor'
                    result = translateBinaryOp(this, "tfTensorListFromTensor", MATLABOutputName, MATLABArgIdentifierNames);
                case 'TensorListGetItem'
                    result = translateNaryOp(this, "tfTensorListGetItem", MATLABOutputName, MATLABArgIdentifierNames);
                case 'TensorListReserve'
                    result = translateTensorListReserve(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);                
                case 'TensorListStack'
                    result = translateNaryOp(this, "tfTensorListStack", MATLABOutputName, MATLABArgIdentifierNames, "tfPack");
                case 'TensorListSetItem'
                    result = translateNaryOp(this, "tfTensorListSetItem", MATLABOutputName, MATLABArgIdentifierNames);
                case 'Tile'
                    result = translateBinaryOp(this, "tfTile", MATLABOutputName, MATLABArgIdentifierNames);
                case 'TopKV2' 
                    result = translateTopKV2(this, node_def, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap);
                case 'Transpose'
                    result = translateTranspose(this, MATLABOutputName, MATLABArgIdentifierNames);         
                case 'Unpack'
                    result = translateUnpack(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'Where'
                    result = translateWhere(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'KerasModelOrLayer'
                    result = translateKerasModelOrLayer(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
                otherwise
                    result = translateUnsupportedOp(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:OperatorNotSupported', MessageArgs={node_def.op}, Placeholder=true, Operator=node_def.op, LayerClass=this.LayerName);
            end
            
            % Generate forward rank code
            if result.ForwardRank
                result.Code = result.Code + newline + MATLABOutputName + "." + this.RANKFIELDNAME + " = " + MATLABArgIdentifierNames{1} + "." + this.RANKFIELDNAME + ";";
            end
        end
        
       code = writeConstant(this, node_def, MATLABOutputName, TFConstants);
       code = writeCapturedInput(this, node_def, MATLABOutputName, TFConstants);
    end
    
    methods (Access = protected)        
        % Generic translators 
        result = translateInlinedUnaryOp(this, outputDims, fcnStr, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateUnaryOp(this, tfOpFcnName, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateBinaryOp(this, tfOpFcnName, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateNaryOp(this, tfOpFcnName, MATLABOutputName, MATLABArgIdentifierNames, tfOpDependencies);
        result = translateInlinedBinaryOp(this, outputDims, fcnStr, isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
        
        % Specialized translators
        result = translateArgMin(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateBiasAdd(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);  
        result = translateCall(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateIdentity(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateMean(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateSelect(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateReadVariableOp(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateRsqrt(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateConcatV2(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translatePack(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateAll(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateTranspose(this, MATLABOutputName, MATLABArgIdentifierNames);        
        result = translateStridedSlice(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateLeakyRelu(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateResizeNearestNeighbor(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateCast(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateFusedBatchNorm(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap); 
        result = translateRelu6(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateMax(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateMaxPool(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateAvgPool(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateConv2D(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateNeg(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateNoOp(this,MATLABOutputName, MATLABArgIdentifierNames);
        result = translateDepthwiseConv2dNative(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translatePad(this, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translatePadV2(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateProd(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateMirrorPad(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateSquaredDifference(this, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translatePow(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateSqueeze(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateSpaceToDepth(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateDepthToSpace(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateIdentityN(this, numOutputs, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateConst(this, node_def, MATLABOutputName, TFConstants);
        result = translateRandomStandardNormal(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateMatMul(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateBatchMatMulV2(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateTopKV2(this, node_def, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap);
        result = translateAssert(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateKerasModelOrLayer(this, nodeDef, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateAssignVariableOp(this);
        result = translateSum(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateResizeBilinear(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateFill(this, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateSlice(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateGather(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateExpandDims(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateUnPack(this, node_def, MATLABOutputNames, MATLABArgIdentifierNames); 
        result = translateWhere(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateNonMaxSuppressionV5(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap);
        result = translateSplit(this, node_def, MATLABOutputName, MATLABArgIdentifierNames); 
        result = translateStatelessIf(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateMin(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateLogicalAnd(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateLessEqual(this, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateTensorListReserve(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateStatelessWhile(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateSquare(this, node_def, MATLABOutputName, MATLABArgIdentifierNames);

        % Placeholder op translator
        result = translateUnsupportedOp(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames); 

        % Translation utilities 
        [MATLABOutputName, MATLABArgIdentifierNames, numOutputs] = prepareNode(this, node_def, nameMap, numOutputMap, multiOutputNameMap, tfModelNodeToModelOut);
        outputNames = makeMultipleOutputArgs(this, outputName, numOutputs);	
		
        function code = writeRankInitializationCode(this, MATLABOutputName, inputRank)
            code = MATLABOutputName + "." + this.RANKFIELDNAME + " = " + num2str(inputRank) + ";"; 
        end
    end
end
