classdef CustomLayerTranslator < handle
    % This class is used to generate the source code of a single custom layer. 

%   Copyright 2021-2024 The MathWorks, Inc.
    
    % Note that this class does not instantiate custom layers or populates
    % them. 
    
    properties (Access = public)
        % Properties used to define the custom layer to generate 
        GraphDef            % A TFGraphDef
        TopFunction         % The TFFunction that this custom layer implements
        CurPlaceholder      % The placeholder layer to be replaced 
        LayerVariables      % A list of all variable names (learnable parameters). These are not yet re-named
        Constants           % A TFConstants to track and name constants used in this custom layer
        OpFunctionsList     % A list of all TensorFlow operator functions used in this layer
        PackageName         % Package where the layer will be saved
        HasUnsupportedOp    % Boolean that is set to true if the custom layer contains an unsupported op
        ImportManager       % ImportManager object reference
        InSubclassed        % Boolean that is set to true if the custom layer is present inside a subclassed model
    end 
    
    properties (Access = public) 
        % Properties used for saving code onto disk
        LayersPath % A string storing the path to put the custom layer
        LayerClass % A string storing the classname of the custom layer
    end
    
    properties (Access = private)
        % Properties used for generating code
        TranslatedFunctionsNameSet % A set that tracks which functions were generated already
        TranslatedFunctions        % A list of FunctionTranslator objects that have been translated
        queue = {}                 % A queue for pseudo-recursive calls
    end
    
    properties (Access = private)
        TemplateLocation = which('templateCustomLayer.txt')
    end 
    
    methods 
        function obj = CustomLayerTranslator(placeholderLayer, graphDef, tffunction, layerVariables, layersPath, layerClass, packageName, importManager)
            import nnet.internal.cnn.tensorflow.*;
            obj.CurPlaceholder = placeholderLayer; 
            obj.GraphDef = graphDef; 
            obj.TopFunction = tffunction;
            obj.LayerVariables = layerVariables; 
            
            obj.LayersPath = layersPath; 
            obj.LayerClass = layerClass; 
            
            obj.TranslatedFunctionsNameSet = containers.Map(); 
            obj.TranslatedFunctions = []; 
            obj.Constants = gcl.TFConstants;
            
            obj.PackageName = packageName;
            obj.ImportManager = importManager;
            
            obj.HasUnsupportedOp = false;
            obj.InSubclassed = false;
        end 
        
        function [legalNames, constantNames] = writeCustomLayer(this)
            % This method will generate a custom layer with a
            % predict method specified by the TFFunction defined by 
            % "TopFunction". All downstream dependencies of the top function 
            % will also be generated. 
            
            % Additionally, this method will write the custom layer
            % onto disk as specified by the LayersPath and LayersClass
            % properties. 
            
            % Finally, the method returns legalized and uniqified constant 
            % names to use. 
            import nnet.internal.cnn.tensorflow.*;
            
            % Translate and emit the forward pass first. 
            this.translateForwardPass() 
            this.populateOpFunctionsList();
            forwardPass = this.writeForwardPass(); 
            
            [nonlearnableParamsCode, learnableParamsCode, legalNames] = this.getCodeForVariables();
            constantNames = this.Constants.getConstNames(); 
            [numLayerInputs, highestOutputIdx, inputArgsCode, layerInputs] = this.getArgsCode(); 
            [extractforwardcall, outputArgsCode] = this.getForwardCallExtractionCode(highestOutputIdx); 

            % Model Inputs Code
            if ~isempty(layerInputs)
                layerInputsCode = strjoin(layerInputs, ", ");
                layerInputsCode = strcat(", ", layerInputsCode);
            else
                layerInputsCode = "";
            end

            % Predict Call
            call = this.writePredictCall(layerInputs, inputArgsCode, outputArgsCode);
            
            % Get template 
            [fid,msg] = fopen(this.TemplateLocation, 'rt');
            % If the support package is correctly installed we expect to
            % find the custom layer template, throw an assertion failure if
            % that is not the case
            assert(fid ~= -1, ['Custom layer template could not be opened: ' msg '. Make sure the support package has been correctly installed.']);
            code = string(fread(fid, inf, 'uint8=>char')'); 
            fclose(fid);

            % Checks for operators that do not support acceleration
            unAcceleratableOps = ["tfCast","tfFloorMod",...
                "tfNonMaxSuppressionV5","tfRandomStandardNormal",...
                "tfSqueeze","tfStatelessIf","tfTopKV2","tfWhere"];
            if isempty(this.OpFunctionsList)
                unacceleratable = false;
            else
                unacceleratable = contains(this.OpFunctionsList, unAcceleratableOps);
            end
            
            % Fill out template
            if ~any(unacceleratable)
                inheritance = "& nnet.layer.Acceleratable"; 
                code = strrep(code, "{{acceleration}}", strjoin(inheritance, newline));
            else
                code = strrep(code, "{{acceleration}}", strjoin("", newline));
            end

            code = strrep(code, "{{autogen_timestamp}}", strjoin(string(datetime('now')), newline));
            code = strrep(code, "{{layerclass}}", this.LayerClass); 
            code = strrep(code, "{{nonlearnables}}", strjoin(nonlearnableParamsCode, newline)); 
            code = strrep(code, "{{learnables}}", strjoin(learnableParamsCode, newline));
            code = strrep(code, "{{literals}}", strjoin(constantNames, newline));
            code = strrep(code, "{{numinputs}}", num2str(numLayerInputs));
            code = strrep(code, "{{inputLabels}}", strcat("{'",strjoin(this.CurPlaceholder.InputLabels,"','"),"'}"));
            code = strrep(code, "{{outputLabels}}", strcat("{'",strjoin(this.CurPlaceholder.OutputLabels,"','"),"'}"));
            code = strrep(code, "{{layerinputs}}", layerInputsCode);
            code = strrep(code, "{{numoutputs}}", num2str(highestOutputIdx));
            code = strrep(code, "{{forwardcall}}", call);
            code = strrep(code, "{{extractforwardcall}}", strtrim(extractforwardcall));
            code = strrep(code, "{{forwardpassdefinition}}", forwardPass);
            code = nnet.internal.cnn.tensorflow.util.indentcode(char(code)); 
            
            % Write generated custom layer to disk
            [fid,msg] = fopen(fullfile(this.LayersPath, this.LayerClass + ".m"), 'w');
            if fid == -1
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnableToCreateCustomLayerFile',this.LayerClass,msg)));
            end
            
            if isempty(this.ImportManager.CustomLayerPath)
                % Set the path to the custom layer package in the import manager, if one was created successfully
                this.ImportManager.CustomLayerPath = this.LayersPath;
            end
            fwrite(fid, code);
            fclose(fid);
        end
    end

    methods (Access = private) 
        
        function [extractforwardcall, outputArgsCode] = getForwardCallExtractionCode(this, highestOutputIdx)
            outputArgsCode = cell(1, highestOutputIdx); 
            extractforwardcall = "";
            for curOut = 1:highestOutputIdx 
                curOutStr = num2str(curOut); 
                outputArgsCode{curOut} = ['temp{' curOutStr '}'];
                prePermutationStr = '';
                postPermutationStr = '';
                if ~this.InSubclassed && this.ImportManager.OnlySupportDlnetwork
                    prePermutationStr = "iPermuteToForwardTF(";
                    postPermutationStr = ", " + ['temp{' curOutStr '}.rank'] + ")";
                end
                % Add output labels inferred by the labeler
                extractforwardcall = extractforwardcall + outputArgsCode{curOut}  + " = " + "addOutputLabel(" ...
                    + [outputArgsCode{curOut} ', ' curOutStr ', obj)'] + ";" + newline;

                % Layer output extract call
                extractforwardcall = extractforwardcall + ['varargout{' curOutStr '}'] + " = " + ...
                    prePermutationStr + ['temp{' curOutStr '}.value'] + postPermutationStr + ";" + newline;
            end  
        end
        
        function [numLayerInputs, highestOutputIdx, inputArgsCode, layerInputs] = getArgsCode(this)
            % Use graph call node to generate the caller input arguments.
            inputargs = this.TopFunction.Signature.input_arg;
            numInputs = numel(this.TopFunction.Signature.input_arg) - 1; % exclude obj argument; 

            inputArgsCode = cell(numInputs + 1, 1);
            layerInputs = {}; 
            curVar = 1; 
            curInput = 1;
            for i = 1:numInputs
                if strcmp(inputargs(i).type, 'DT_RESOURCE')
                    curLegalVarName = this.LayerVariables{curVar}.legalName; 
                    inputArgsCode{i} = "obj." + (curLegalVarName);
                    curVar = curVar + 1; 
                else
                    inputArgsCode{i} = inputargs(i).name;
                    layerInputs{curInput} = inputargs(curInput).name; %#ok<AGROW> 
                    curInput = curInput + 1; 
                end
            end
            inputArgsCode{end} = 'obj'; 

            % Use signature to generate the output args and extraction code
            highestOutputIdx = this.CurPlaceholder.NumOutputs; 
            numLayerInputs = curInput - 1;            
        end

        function call = writePredictCall(this, layerInputs, inputArgsCode, outputArgsCode)              
            call = string(['import ' this.PackageName '.ops.*;']);
                for i=1:numel(layerInputs)
                    call = call + newline + strjoin(["obj.InputLabels{",string(i),"} = ", layerInputs{i}, ".dims;"],"") + newline;
                    inputRank = num2str(numel(this.TopFunction.attr.x_input_shapes.list.shape(i).dim));
                    if this.ImportManager.OnlySupportDlnetwork && ~this.InSubclassed
                        % Permute from labeled forward TF to unformatted reverse TF
                        call = call + newline + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("iPermuteToReverseTF", layerInputs{i}, {layerInputs{i}, inputRank}) + newline;
                    else
                        % Permute from labeled reverse TF to unformatted reverse TF
                        call = call + newline + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("iPermuteToReverseTF", layerInputs{i}, {layerInputs{i}, inputRank, "true"}) + newline;
                    end
                end
            call = call + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(this.TopFunction.Signature.legalname, outputArgsCode, inputArgsCode);
        end
        
        function [nonlearnableParamsCode, learnableParamsCode, legalNames] = getCodeForVariables(this)
            import nnet.internal.cnn.tensorflow.*;
            
            layerVariableNames = cellfun(@(x)(string(x.curVarName)), this.LayerVariables);
            
            if ~isempty(layerVariableNames)
                legalNames = gcl.util.iMakeLegalMATLABNames(layerVariableNames);
            else 
                legalNames = []; 
            end

            nonlearnableParamsCode = string.empty;
            learnableParamsCode = string.empty; 
            for i = 1:numel(this.LayerVariables) 
                if this.LayerVariables{i}.IsLearnable
                    learnableParamsCode(end + 1) = legalNames(i); %#ok<AGROW>
                else 
                    nonlearnableParamsCode(end + 1) = legalNames(i); %#ok<AGROW>
                end
                this.LayerVariables{i}.legalName = legalNames(i); 
            end
        end
        
        function translateForwardPass(this) 
            import nnet.internal.cnn.tensorflow.*;
            topFcnTranslator = gcl.FunctionTranslator(this.TopFunction, this.Constants, true); 
            topFcnTranslator.translateFunction(this.ImportManager, this.LayerClass);
            this.TranslatedFunctions = topFcnTranslator;
            this.TranslatedFunctionsNameSet(this.TopFunction.Signature.name) = 1; 
            this.queue = topFcnTranslator.SubFunctions; 
            if ~this.HasUnsupportedOp && topFcnTranslator.HasUnsupportedOp
                this.HasUnsupportedOp = true;
            end            
            % Generate subfunctions (dependencies)
            this.translateSubFunctions(); 
        end

        function translateSubFunctions(this)
            % FIFO re-calling to generate function dependencies of
            % previously generated function.
            import nnet.internal.cnn.tensorflow.*;
            while ~isempty(this.queue)
                % pop from queue
                subfcnname = this.queue{1}; 
                this.queue(1) = []; 
                
                % only generate the function if it has not been seen before
                if ~isKey(this.TranslatedFunctionsNameSet, subfcnname)
                    this.TranslatedFunctionsNameSet(subfcnname) = 1; 
                    
                    subfcntranslator = gcl.FunctionTranslator(this.GraphDef.findFunction(subfcnname), this.Constants, false); 
                    subfcntranslator.translateFunction(this.ImportManager, this.LayerClass);
                    this.TranslatedFunctions = [this.TranslatedFunctions subfcntranslator]; 
                    this.queue = [this.queue subfcntranslator.SubFunctions]; 
                    if ~this.HasUnsupportedOp && subfcntranslator.HasUnsupportedOp
                        this.HasUnsupportedOp = true;
                    end 
                end
            end
        end
        
        function populateOpFunctionsList(this)
            for curFunction = 1:numel(this.TranslatedFunctions)
                curTranslatedFunction = this.TranslatedFunctions(curFunction);
                for i = 1:numel(curTranslatedFunction.NodeTranslations)
                    this.OpFunctionsList = [this.OpFunctionsList curTranslatedFunction.NodeTranslations(i).OpFunctions]; 
                end
            end
        end
        
        function code = writeForwardPass(this)
            % Generate the entire forward pass of this custom layer into a
            % string. 
            code = ""; 
            opsPackageImportStr = ['import ' this.PackageName '.ops.*;'];
            for curFunction = 1:numel(this.TranslatedFunctions)
                code = code + this.TranslatedFunctions(curFunction).emitFunction(opsPackageImportStr); 
            end
        end

    end
end 
