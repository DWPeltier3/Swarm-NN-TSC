classdef ImportManager < handle
    %ImportManager Class that dispatches TensorFlow model import

    %   Copyright 2023 The MathWorks, Inc                
    
    properties
        ModelPath
        PackageName
        ImageInputSize 
        TargetNetwork 
        Verbose
        OutputLayerType = '';
        Classes = '';
        OnlySupportDlnetwork = false;
        ImportIssues = []; % for tracking translation issues
        ImportedInputs = {};
        CustomLayerPath = '';
        CallFromSDK
    end
    
    methods
        function obj = ImportManager(ModelPath, options, CallFromSDK)
            if nargin < 3
                CallFromSDK = false;                
            end
            
            obj.ModelPath = ModelPath;
            obj.PackageName = options.PackageName;
            obj.ImageInputSize = options.ImageInputSize;
            obj.TargetNetwork = options.TargetNetwork;
            obj.Verbose = options.Verbose;
            obj.CallFromSDK = CallFromSDK;
            obj.OutputLayerType = '';
            obj.Classes = '';
            
            if isfield(options,'OutputLayerType')
                obj.OutputLayerType = options.OutputLayerType;
            end
            if isfield(options,'Classes')
                obj.Classes = options.Classes;
            end            
            if isfield(options,'OnlySupportDlnetwork')
                obj.OnlySupportDlnetwork = options.OnlySupportDlnetwork;
            end
        end

        function [net] = translate(this)
            import nnet.internal.cnn.tensorflow.*;   
            
            this.validatePackageName;
            this.validateTFModelFolder;
            
            nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseImportStarts', this.Verbose);
            this.ImageInputSize = nnet.internal.cnn.keras.util.validateImageInputSize(this.ImageInputSize);
            sm = savedmodel.TFSavedModel(this.ModelPath, this, true); 

            if strcmp(this.TargetNetwork, 'dlnetwork') && nnet.internal.cnn.tensorflow.util.hasFoldingLayer(sm.KerasManager.LayerGraph)        
                if ~this.OnlySupportDlnetwork
                    % Add message about using 'TargetNetwork' as 'DAGNetwork' only for
                    % old TensorFlow API which supports import to DAGNetwork
                    error(message('nnet_cnn_kerasimporter:keras_importer:SequenceFoldingFoundWithDlnetwork'));
                end 
            end
            
            nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseTranslationStarts', this.Verbose);    
            % parse inputs
            
            if ismember(sm.KerasManager.RootClassType, {'_generic_user_object'})
                % Subclassed model detected
                net = nnet.internal.cnn.tensorflow.importSubclassedModel(sm);
            elseif ismember(sm.KerasManager.RootClassType, {'_tf_keras_model', '_tf_keras_network', '_tf_keras_sequential'}) && isempty(sm.KerasManager.LayerGraph)
                % Saved using low-level SavedModelAPI
                net = nnet.internal.cnn.tensorflow.importSubclassedModel(sm);
            else
                % Sequential or Functional model detected
                if isempty(this.PackageName)
       		        [layers, hasUnsupportedOp, supportsInputLayers, inputShapes] = gcl.translateTFKerasLayers(sm); 
                else 
	                [layers, hasUnsupportedOp, supportsInputLayers, inputShapes] = gcl.translateTFKerasLayers(sm, this.PackageName);
                end

                % Set the input shape and format information
                this.ImportedInputs = inputShapes;            
                nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseAssemblingStarts', this.Verbose);
                [~,id] = findPlaceholderLayers(layers);
            
                if any(id)
                    % If imported network has any placeholder layers
                    net = dlnetwork(layers, 'Initialize', false);
                    this.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UninitDlnetworkWithPlaceholder');
                elseif hasUnsupportedOp
                    % If imported network has any unsupported ops in custom layers
                    net = dlnetwork(layers, 'Initialize', false);
                    this.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:GeneratedDlnetContainsUnsupportedOpWarning');                    
                else
                    [layerGraph, minLengthRequired] = nnet.internal.cnn.keras.util.checkMinLengthRequired(layers);
                    if minLengthRequired
                        layers = nnet.internal.cnn.keras.util.autoSetMinLength(layerGraph);
                    end
                    if strcmp(this.TargetNetwork, 'dagnetwork')
                        if supportsInputLayers
                            % Check if the input layer is supported by
                            % DAGNetworks (generic inputLayer is not supported)
                            layers = nnet.internal.cnn.keras.util.configureOutputLayerForNetwork(layers, this.Classes);
                            net = assembleNetwork(layers);
                        else
                            if sm.KerasManager.AM.isTimeDistributed
                                % Cannot create a dlnetwork since the model contains sequence folding-unfolding layers
                                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:SequenceFoldingFoundWithDAGNetworkWithoutInput'))); 
                            else 
                                this.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnsupportedImportAsDAGNetwork');
                                
                                % Since import as 'DAGNetwork' is not possible change
                                % targetNetwork type to 'dlnetwork' and remove the output
                                % layer/ layers
                                this.TargetNetwork = 'dlnetwork';
                                layers = removeLayers(layers, layers.OutputNames);
                            end
                        end
                    end
                    if strcmp(this.TargetNetwork, 'dlnetwork')
                        if ~isempty(this.OutputLayerType)
                            this.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:OutputLayerTypeNotNeededForDlnetwork');                            
                        end
                        if supportsInputLayers         
                            net = dlnetwork(layers, 'Initialize', true); 
                        else
                            % Check if input data size has NaNs in it
                            if isInit(inputShapes)
                                dlX = createExampleInputs(inputShapes);
                                try
                                    net = dlnetwork(layers, dlX{:});
                                    % Display how to initialize all inputs
                                    this.throwInitializeWarning(net.InputNames, inputShapes, true);
                                catch
                                    fmt = extractInputFormats(inputShapes);
                                    net = dlnetwork(layers, 'Initialize', false);
                                    this.throwInitializeWarning(net.InputNames, fmt, false);
                                end
                            else
                                % Throw warning that the input shape is not defined use
                                fmt = extractInputFormats(inputShapes);
                                net = dlnetwork(layers, 'Initialize', false);
                                % Display input rank and shape information
                                this.throwInitializeWarning(net.InputNames, fmt, false);
                            end
                        end
                    end
                end
            end
            nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseImportFinished', this.Verbose);
        end

        function validatePackageName(this)
            this.PackageName = char(this.PackageName);
            if ~isWritable
                this.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:FolderNotWritable', MessageArgs={pwd});                
            end
        end

        function validateTFModelFolder(this)  
            this.ModelPath = char(this.ModelPath);
            folderExists = isfolder(this.ModelPath);
            fileExists = isfile(this.ModelPath);
            if ~folderExists && ~fileExists
                % neither a folder nor a file exists at the given path
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FolderNotFound', this.ModelPath)));
            elseif ~folderExists && fileExists
                % a folder does not exist but a file exists at the given path
                % check if trying to import an older keras HDF5 model file
                maybeKerasFormat = nnet.internal.cnn.tensorflow.util.checkKerasFileFormat(this.ModelPath);
                if maybeKerasFormat
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FolderNotFoundButH5FileExistsImportLayers', this.ModelPath)));
                else
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FolderNotFound', this.ModelPath)));
                end
            elseif folderExists
                % a folder exists, we should check if there is a 'saved_model.pb' file and a 'variables' sub folder
                if ~isfile([this.ModelPath filesep 'saved_model.pb'])
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:SavedModelPbNotFound', this.ModelPath)));
                end
                if ~isfolder([this.ModelPath filesep 'variables'])
                    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:VariablesNotFound', this.ModelPath)));
                end
            end
        end

        function throwInitializeWarning(this, inputNames, input, isInit)
            % Throws a warning to guide the user to add input layers or initalize the
            % imported dlnetwork.
            if isInit
                % Initialized dlnetwork
                if this.OnlySupportDlnetwork
                    % Update initialize message to forward TensorFlow format
                    % ordering if the call is from importNetworkFromTensorFlow
                    prologue = getString(message("nnet_cnn_kerasimporter:keras_importer:PrologueInitDlnetworkNewAPI"));
                else
                    prologue = getString(message("nnet_cnn_kerasimporter:keras_importer:PrologueInitDlnetwork"));
                end
                sampleCode = nnet.internal.cnn.tensorflow.util.sampleCodeToInitialize(inputNames, input, isInit); 
                epilogue = getString(message("nnet_cnn_kerasimporter:keras_importer:DlarrayFormatInfo")); 
            else
                % Unitialized dlnetwork
                if ismember(input, {'SSCB', 'SSSCB', 'CB', 'CBT', 'SCBT', 'SSCBT', 'SSSCBT'})
                    prologue = getString(message("nnet_cnn_kerasimporter:keras_importer:PrologueAddInputLayers"));
                    sampleCode = nnet.internal.cnn.tensorflow.util.sampleCodeToAddInputLayers(inputNames, input);
                    epilogue = getString(message("nnet_cnn_kerasimporter:keras_importer:DlarrayFormatInfo"));
                else
                    prologue = getString(message("nnet_cnn_kerasimporter:keras_importer:PrologueUnInitDlnetwork"));
                    sampleCode = nnet.internal.cnn.tensorflow.util.sampleCodeToInitialize(inputNames, input, isInit);
                    epilogue = getString(message("nnet_cnn_kerasimporter:keras_importer:DlarrayFormatInfo"));  
                end
            end
        
            instructions = join([sampleCode, epilogue], [newline newline]);
            if ~ismissing(instructions)
                msg = join([prologue, instructions], [newline newline]);
                this.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:InputSizeAndFormatInfoForDlnetwork', MessageArgs={msg});                
            end
        end

        function createAndProcessImportIssue(this, args)
            arguments
                this
                args.Operator = string.empty;
                args.MessageID = string.empty;
                args.MessageArgs = {};
                args.Placeholder = false;
                args.LayerClass = "";
            end
            issue = nnet.internal.cnn.keras.ImportIssue(Operator=args.Operator,MessageID=args.MessageID,MessageArgs=args.MessageArgs,Placeholder=args.Placeholder,LayerClass=args.LayerClass);
            this.processImportIssue(issue);
        end

        function processImportIssue(this, issue)
            if this.CallFromSDK
                % Keep track of the issue without displaying a warning
                this.ImportIssues = [this.ImportIssues issue];
            else
                % display warning message only
                issue.warningWithoutBacktrace;
            end
        end 
    end
    
end

function tf = isInit(inputShapes)
    X = cellfun(@(x) any(isnan(x{1})), inputShapes, 'UniformOutput', true);
    tf = ~any(X);
end

function X = createExampleInputs(inputShapes)
    X = cellfun(@(x) networkDataLayout(x{1}, x{2}), inputShapes, 'UniformOutput', false);
end

function X = extractInputFormats(inputShapes)
    X = cellfun(@(x) nnet.internal.cnn.tensorflow.util.sortToDLTLabel(x{2}), inputShapes, 'UniformOutput', false);
end

function writeStatus = isWritable
% Checks if current directory is writable
    tempDir = strsplit(tempname, filesep);
    tempDir = tempDir{end};
    [writeStatus, ~] = mkdir(tempDir); 
    if writeStatus
        rmdir(tempDir);
    end
end
