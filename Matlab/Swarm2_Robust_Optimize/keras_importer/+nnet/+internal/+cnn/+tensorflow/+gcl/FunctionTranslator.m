classdef FunctionTranslator < handle
%

%   Copyright 2021-2023 The MathWorks, Inc.

    properties
        Function          % A TFFunction 
        NodeTranslations  % A list of NodeTranslationResult objects 
        SubFunctions      % A list of subfunctions that this function calls
        IsTopCall         % A boolean that signals whether or not this class is the top level function of a custom layer
        NameMap           % A map that stores tensorflow names to legalized matlab names
        Constants         % A TFConstants
        HasUnsupportedOp  % Boolean that is set to true if the function contains an unsupported ops
    end
    
    properties (Constant) 
        RANKFIELDNAME = "rank"; 
    end 
    
    methods
        function obj = FunctionTranslator(tffunction, constants, isTopCall)
            import nnet.internal.cnn.tensorflow.*;
            obj.Function = tffunction; 
            obj.Constants = constants;
            obj.IsTopCall = isTopCall;
            obj.NodeTranslations = gcl.NodeTranslationResult.empty;
            obj.HasUnsupportedOp = false;
        end
        
        function translateFunction(this, importManager, layerName)
            import nnet.internal.cnn.tensorflow.*;
            translationStrategy = gcl.translators.tfOpTranslator;
            translationStrategy.ImportManager = importManager;
            translationStrategy.LayerName = layerName;
            this.NameMap = this.Function.tfNameToMATLABName;
            
            % Tracks the multi-output ops (name->number of outputs) 
            numOutputMap = containers.Map();

            % Tracks input names associated with multi-output ops (name-> List of input names) 
            multiOutputNameMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            
            this.Function.Signature.convertToMATLABNames(this.NameMap);
            
            for i = 1:numel(this.Function.node_def)
                curnode = this.Function.node_def(i); 
                if isempty(curnode)
                    % input 
                else
                    this.NodeTranslations(end + 1) = translationStrategy.translateTFOp(curnode, this.NameMap, this.Constants, numOutputMap, multiOutputNameMap, this.Function.tfModelNodeToModelOut); 
                    if ~this.HasUnsupportedOp && ~this.NodeTranslations(end).Success
                        this.HasUnsupportedOp = true;
                    end
                    if ~isempty(this.NodeTranslations(end).SubFunctions)
                        for j = 1:numel(this.NodeTranslations(end).SubFunctions)
                            this.SubFunctions{end + 1} = this.NodeTranslations(end).SubFunctions{j};
                        end
                    end
                end
            end
        end
        
        function code = emitFunction(this, opsPackageImportStr)
            % Used for writing gcl in functional sequential models or
            % top-level layer in sub-classed models
            code = this.Function.Signature.writeMATLABSignature();
            % add ops package import
            code = code + opsPackageImportStr + newline;
            if this.IsTopCall
                code = code + this.Function.writeInputRankTracker(this.RANKFIELDNAME);
            end

            for i = 1:numel(this.NodeTranslations)
                code = code + this.NodeTranslations(i).emitNode(); 
            end 
            
            code = code + this.Function.Signature.writeOutputAssignCode(this.Function.ret, this.NameMap);
            code = code + "end" + string(newline) + string(newline);
        end 

        function code = emitFunction_v2(this, opsPackageImportStr)
            % Used for writing gcl for nested layers and functional models 
            % in subclassed models
            code = this.Function.Signature.writeMATLABSignature();
            % add ops package import
            code = code + opsPackageImportStr + newline;
            if this.IsTopCall
                code = code + this.Function.writeInputRankTracker_v2(this.RANKFIELDNAME);
            end
            
            for i = 1:numel(this.NodeTranslations)
                code = code + this.NodeTranslations(i).emitNode(); 
            end 

            code = code + this.Function.Signature.writeOutputAssignCode(this.Function.ret, this.NameMap);
            code = code + "end" + string(newline) + string(newline); 
        end
        
    end
end
