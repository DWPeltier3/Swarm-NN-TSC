classdef tfOpLambdaTranslator < nnet.internal.cnn.tensorflow.gcl.translators.tfOpTranslator
    % tfOpLambdaTranslator: This class is used to create NodeTranslationResults
    % from TFOpLambda Placeholder Layer

%   Copyright 2023 The MathWorks, Inc.
    
    methods (Access = public)
        function result = translateTFOpLambdaPlaceholder(this, tfOpLambdaLayer)
            import nnet.internal.cnn.keras.util.*;									  
            kerasConfig = tfOpLambdaLayer.KerasConfiguration;

            tfOpLambdaFcn = kerasConfig.function;
            layerInbndNodes = kerasConfig.inbound_nodes{1};
            MATLABOutputName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({kerasConfig.name});
            MATLABInputsStruct = getFunctionInputs(this, layerInbndNodes, tfOpLambdaLayer.InputNames);
            [inputCode, MATLABArgIdentifierNames] = processFunctionInputs(this, MATLABInputsStruct);
            [outputCode] = processFunctionOutputs(this, tfOpLambdaLayer.NumOutputs, MATLABOutputName);
            
            switch tfOpLambdaFcn
                %tf functions
                case 'cast'
                    result = translateBinaryOp(this, "tfCast", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'clip_by_value'
                    result = translateNaryOp(this, "tfClipByValue", MATLABOutputName, MATLABArgIdentifierNames);
                case 'compat.v1.squeeze'
                    result = translateBinaryOp(this, "tfSqueeze", MATLABOutputName, MATLABArgIdentifierNames);
                case 'concat'
                    result = translateConcatV2(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'expand_dims'
                    result = translateExpandDims(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'fill'
                    result = translateFill(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'realdiv'
                    result = translateBinaryOp(this, "tfDiv", MATLABOutputName, MATLABArgIdentifierNames);
                case 'reshape'
                    result = translateNaryOp(this, "tfReshape", MATLABOutputName, MATLABArgIdentifierNames);
                case 'shape'
                    result = translateUnaryOp(this, "tfShape", MATLABOutputName, MATLABArgIdentifierNames);
                case 'size'
                    result = translateUnaryOp(this, "tfSize", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'slice'
                    result = translateSlice(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'stop_gradient'
                    result = translateUnaryOp(this, "tfStopGradient", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'tile'
                    result = translateBinaryOp(this, "tfTile", MATLABOutputName, MATLABArgIdentifierNames);
                
                %tf.math functions
                case {'math.add', '__operators__.add'}
                    result = translateBinaryOp(this, "tfAdd", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'math.add_n'
                    result = translateNaryOp(this, "tfAddN", MATLABOutputName, MATLABArgIdentifierNames, "tfAdd");
                case 'math.argmin'
                    result = translateBinaryOp(this, "tfArgMin", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.truediv'
                    result = translateBinaryOp(this, "tfDiv", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.equal'
                    isLogicalOp = true;
                    result = translateInlinedBinaryOpLambdaOp(this, "eq", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'math.exp'
                    result = translateInlinedUnaryOpLambdaOp(this, "exp", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.floordiv'
                    result = translateBinaryOp(this, "tfFloorDiv", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'math.floormod'
                    result = translateBinaryOp(this, "tfFloorMod", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.multiply'
                    result = translateBinaryOp(this, "tfMul", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.greater'
                    isLogicalOp = true;
                    result = translateInlinedBinaryOpLambdaOp(this, "gt", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames); 
                case 'math.greater_equal'
                    isLogicalOp = true;
                    result = translateInlinedBinaryOpLambdaOp(this, "ge", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.less'
                    isLogicalOp = true;
                    result = translateInlinedBinaryOpLambdaOp(this, "lt", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.log'
                    result = translateInlinedUnaryOpLambdaOp(this, "log", MATLABOutputName, MATLABArgIdentifierNames); 
                case 'math.maximum'
                    result = translateBinaryOp(this, "tfMaximum", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.minimum'
                    result = translateBinaryOp(this, "tfMinimum", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.negative'
                    result = translateNeg(this, MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.pow'
                    isLogicalOp = true;
                    result = translateInlinedBinaryOpLambdaOp(this, "power", isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.reduce_all'
                    result = translateNaryOp(this, "tfAll", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.reduce_prod'
                    result = translateNaryOp(this, "tfProd", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.reduce_sum'
                    result = translateNaryOp(this, "tfSum", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.sigmoid'
                    result = translateInlinedUnaryOpLambdaOp(this, "sigmoid", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.subtract' 
                    result = translateBinaryOp(this, "tfSub", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.tanh'
                    result = translateInlinedUnaryOpLambdaOp(this, "tanh", MATLABOutputName, MATLABArgIdentifierNames);
                case 'math.top_k'
                    result = translateMOBinaryOp(this, "tfTopKV2", tfOpLambdaLayer.NumOutputs, MATLABOutputName, MATLABArgIdentifierNames);                
                case 'math.sqrt'
                    result = translateInlinedUnaryOpLambdaOp(this, "sqrt", MATLABOutputName, MATLABArgIdentifierNames);
                otherwise
                    validOpName =  nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames({tfOpLambdaFcn});
                    OpLambdaNode.op = validOpName{1};
                    result = translateUnsupportedOp(this, OpLambdaNode, tfOpLambdaLayer.NumOutputs, MATLABOutputName, MATLABArgIdentifierNames);
                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:OperatorNotSupported', MessageArgs={OpLambdaNode.op}, Placeholder=true, Operator=OpLambdaNode.op, LayerClass=this.LayerName);
            end
            
            % Prepend constant assignment input code, if any
            if ~(inputCode == "")
                result.Code = inputCode + newline + result.Code + newline;
            end
            
            % Generate forward rank code
            if result.ForwardRank
                result.Code = result.Code + newline + MATLABOutputName + "." + this.RANKFIELDNAME + " = " + "ndims(" + MATLABOutputName + ".value);";
            end
            
            % Append labeling and post-processing code
            result.Code = result.Code + newline + outputCode;
        end
    end
    
    methods (Access = protected)        
        % Generic translators specific to TFOpLambda layers
        result = translateMOBinaryOp(this, tfOpFcnName, numOutputs, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateInlinedBinaryOpLambdaOp(this, fcnStr, isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames);
        result = translateInlinedUnaryOpLambdaOp(this, fcnStr, MATLABOutputName, MATLABArgIdentifierNames);
    end
end
