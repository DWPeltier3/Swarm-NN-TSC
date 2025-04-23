classdef TFOpLambdaLayerTranslator < handle
    % This class is used to generate the source code 
    % of a custom layer for a TFOpLambda layer. 
    
    %   Copyright 2023-2024 The MathWorks, Inc.
    
    properties (Access = public)
        % Properties used to define the custom layer to generate 
        CurPlaceholder      % The placeholder layer to be replaced 
        Constants           % A TFConstants to track and name constants used in this custom layer
        OpFunctionsList     % A list of all TensorFlow operator functions used in this layer
        PackageName         % Package where the layer will be saved
        HasUnsupportedOp    % Boolean that is set to true if the custom layer contains an unsupported op
        ImportManager       % ImportManager object reference
    end 
    
    properties (Access = public) 
        % Properties used for saving code onto disk
        LayersPath % A string storing the path to put the custom layer
        LayerClass % A string storing the class name of the custom layer
    end
    
    properties (Access = private)
        TemplateLocation = which('templateCustomLayer.txt')
    end 
    
    methods 
        function obj = TFOpLambdaLayerTranslator(placeholderLayer, layersPath, layerClass, packageName, importManager)
            import nnet.internal.cnn.tensorflow.*;
            
            obj.CurPlaceholder = placeholderLayer; 
            obj.Constants = gcl.TFConstants;            
            obj.PackageName = packageName;
            obj.HasUnsupportedOp = false;
            obj.ImportManager = importManager;
            
            obj.LayersPath = layersPath; 
            obj.LayerClass = layerClass;             
        end 
        
        function writeCustomLayer(this)
            % This method will generate a custom layer with
            % predict and forward methods for the 'function' 
            % defined in the TFOpLambda layer's KerasConfiguration
            
            % Additionally, this method will write the custom layer
            % onto disk as specified by the LayersPath and LayersClass
            % properties. 
            
            % Finally, the method returns legalized and uniqified constant 
            % names to use. 
            import nnet.internal.cnn.tensorflow.*;
            
            % Translate and emit the forward pass first. 

            translationStrategy = nnet.internal.cnn.tensorflow.gcl.translators.tfOpLambdaTranslator;
            translationStrategy.ImportManager = this.ImportManager;
            translationStrategy.LayerName = this.LayerClass;
            
            TFOpLambdaNodeTranslation = translationStrategy.translateTFOpLambdaPlaceholder(this.CurPlaceholder); 
            if ~this.HasUnsupportedOp && ~TFOpLambdaNodeTranslation.Success
                this.HasUnsupportedOp = true;
            end
            this.OpFunctionsList = [this.OpFunctionsList TFOpLambdaNodeTranslation.OpFunctions];
            
            call = string(['import ' this.PackageName '.ops.*;']);
            call = call + newline + TFOpLambdaNodeTranslation.Code;            
           
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
            code = strrep(code, "{{nonlearnables}}", ""); 
            code = strrep(code, "{{learnables}}", "");
            code = strrep(code, "{{literals}}", "");
            code = strrep(code, "{{numinputs}}", num2str(this.CurPlaceholder.NumInputs));
            code = strrep(code, "{{inputLabels}}", strcat("{'",strjoin(this.CurPlaceholder.InputLabels,"','"),"'}"));
            code = strrep(code, "{{outputLabels}}", strcat("{'",strjoin(this.CurPlaceholder.OutputLabels,"','"),"'}"));
            code = strrep(code, "{{layerinputs}}", strcat(",", strjoin(this.CurPlaceholder.InputNames,",")));
            code = strrep(code, "{{numoutputs}}", num2str(this.CurPlaceholder.NumOutputs));
            code = strrep(code, "{{forwardcall}}", call);
            code = strrep(code, "{{extractforwardcall}}", "");
            code = strrep(code, "{{forwardpassdefinition}}", "");
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
end 
