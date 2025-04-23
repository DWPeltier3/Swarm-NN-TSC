classdef ExportNetworkToTensorFlowImpl

    %   Copyright 2022 The MathWorks, Inc.

    properties (Constant)
        PythonVersionsTested = ["3.8.12"];
        TensorflowVersionsTested = ["2.6.0"];
        % To find the current python and TF versions in python:
        %     print("PYTHON VERSION:")
        %     import sys
        %     print(sys.version)
        %     vi = sys.version_info
        %     print(vi)
        %     print(vi[0])
        %     print(vi[1])
        %     print(vi[2])
        %
        %     print("TENSORFLOW VERSION:")
        %     import tensorflow
        %     print(tensorflow.__version__)
    end

    properties
        net                 % A SeriesNetwork, DAGNetwork, or initialized dlnetwork
        packagePath         string
        InitialWarnings     message
    end

    methods
        function this = ExportNetworkToTensorFlowImpl(net, packagePath)
            % Check and standardize args
            if isempty(net) || ...
                    ~(isa(net, "SeriesNetwork") || isa(net, "DAGNetwork") || isa(net, "dlnetwork") || isa(net, "nnet.cnn.LayerGraph") || isa(net, "nnet.cnn.layer.Layer"))
                throwAsCaller(MException(message("nnet_cnn_kerasimporter:keras_importer:exporterWrongInputType")));
            end
            if ischar(packagePath)
                packagePath = string(packagePath);
            end
            if isempty(packagePath) || ~isstring(packagePath) || ~isscalar(packagePath) || packagePath==""
                throwAsCaller(MException(message("nnet_cnn_kerasimporter:keras_importer:exporterWrongInputType2")));
            end                
            [this.net, this.packagePath, this.InitialWarnings] = iConvertArgs(net, packagePath);
            % Register message catalog
            nnet.internal.cnn.keras.setAdditionalResourceLocation();
        end

        function doExport(this)
            % Check package name and path
            [parentDir, packageName, Extension] = fileparts(this.packagePath);
            if strlength(Extension)>0
                error(message("nnet_cnn_kerasimporter:keras_importer:exporterPackageNameHasExtension", Extension));
            end
            if strlength(parentDir)>0 && ~exist(parentDir, 'dir')
                error(message("nnet_cnn_kerasimporter:keras_importer:exporterFilepathDoesNotExist", parentDir));
            end
            % Convert network
            networkConverter = nnet.internal.cnn.tf_exporter.NetworkConverter(this.net);
            convertedNetwork = networkToTensorflow(networkConverter, this.InitialWarnings);
            % write output files
            modelFileName  = fullfile(this.packagePath, "model.py");
            initFileName   = fullfile(this.packagePath, "__init__.py");
            weightFileName = fullfile(this.packagePath, "weights.h5");
            READMEFileName = fullfile(this.packagePath, "README.txt");
            % Write the python and README text files
            writeTextFiles(convertedNetwork, parentDir, packageName, modelFileName, initFileName, READMEFileName);
            % Write the binary weight file
            writeWeightFile(convertedNetwork, weightFileName);
        end
    end
end

function [network, packagePath, initialWarnings] = iConvertArgs(network, packagePath)
% Convert an input layer graph, layer array, or uninitialized dlnetwork
% into an exportable network
[network, initialWarnings] = iConvertToNetworkIfNeeded(network);
% If packagePath is just a word, add ./ to the front to avoid
% 'exist' searching the entire PATH for it.
[FILEPATH,NAME,EXT] = fileparts(packagePath); %#ok<ASGLU>
if strlength(FILEPATH) == 0
    packagePath = fullfile("./", packagePath);
end
packagePath = string(packagePath);
end

function [obj, warnings] = iConvertToNetworkIfNeeded(obj)
warnings = message.empty;
% Try to convert the obj to a DAGNetwork or initialized dlnetwork. warnings
% is an array of message objects.
if isa(obj, "SeriesNetwork") || isa(obj, "DAGNetwork") || isa(obj, "dlnetwork") && obj.Initialized
    % It's already an (initialized) network. No conversion needed.
    return;
end
if isa(obj, "dlnetwork") && ~obj.Initialized
    % It's an uninitialized dlnetwork. Try to initialize it. If that fails,
    % extract its layerGraph and continue.
    try
        obj = initialize(obj);
        if obj.Initialized
            return;
        end
    catch
    end
    obj = layerGraph(obj);
end
if isa(obj, "nnet.cnn.LayerGraph") || isa(obj, "nnet.cnn.layer.Layer")
    % Make the LG or LA a layerGraph
    if isa(obj, "nnet.cnn.layer.Layer")
        lg = layerGraph(obj);
    else
        lg = obj;
    end
    % Try to convert the LG to DAGNetwork
    try
        S = warning('OFF', 'nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasWarnings');
        C = onCleanup(@()warning(S));
        obj = assembleNetwork(lg);
        clear C
        return;
    catch
    end
    % Try to convert the LG to initialized dlnetwork
    try
        obj = dlnetwork(lg);
        if obj.Initialized
            return;
        end
    catch
    end
    % Neither of those LG conversions worked. Remove output layers and
    % empty input normalizations and try to make it an initialized
    % dlnetwork
    [lg, modifiedInputLayerNames] = iRemoveEmptyInputNormalizations(lg);
    [lg, newOutputNames] = iRemoveOutputLayers(lg);
    try
        if ~isempty(newOutputNames)
            obj = dlnetwork(lg, OutputNames=newOutputNames(:)');    % Pass the new names for the existing output tensors.
        else
            obj = dlnetwork(lg);                            % Let dlnetwork determine the outputs by default.
        end
        if obj.Initialized
            warnings = [warnings, iMaybeWarnaboutInputNorms(modifiedInputLayerNames)];
            return;
        end
    catch
    end
    % No attempt to convert the input into a DAGNetwork or initialized
    % dlnetwork succeeded
    throwAsCaller(MException(message("nnet_cnn_kerasimporter:keras_importer:exporterFailedToConvertLG")));
end
end

function [lg, modifiedInputLayerNames] = iRemoveEmptyInputNormalizations(lg)
% If any input layers have normalizations with empty parameters, replace
% the layers with equivalent input layers that have no normalization.
modifiedInputLayerNames = string.empty;
for i=1:numel(lg.Layers)
    layer = lg.Layers(i);
    constructor = [];
    if isa(layer, 'nnet.cnn.layer.ImageInputLayer') && layer.Normalization~="none" && iHasEmptyNormalization(layer)
        constructor = @imageInputLayer;
    elseif isa(layer, 'nnet.cnn.layer.Image3dInputLayer') && layer.Normalization~="none" && iHasEmptyNormalization(layer)
        constructor = @image3dInputLayer;
    elseif isa(layer, 'nnet.cnn.layer.SequenceInputLayer') && layer.Normalization~="none" && iHasEmptyNormalization(layer)
        constructor = @sequenceInputLayer;
    elseif isa(layer, 'nnet.cnn.layer.FeatureInputLayer') && layer.Normalization~="none" && iHasEmptyNormalization(layer)
        constructor = @featureInputLayer;
    end
    if ~isempty(constructor)
        layer = constructor(layer.InputSize, Name=layer.Name, Normalization="none");
        lg = replaceLayer(lg, layer.Name, layer);
        modifiedInputLayerNames(end+1) = string(layer.Name); %#ok<AGROW>
    end
end
end

function tf = iHasEmptyNormalization(inputLayer)
tf = isempty(inputLayer.Min) && isempty(inputLayer.Max) && isempty(inputLayer.Mean) ...
    && isempty(inputLayer.StandardDeviation);
end

function [lg, newOutputNames] = iRemoveOutputLayers(lg)
% If there are any official output layers, remove them, but preserve the
% tensors they represent. This means change the OutputNames of the lg to
% the names of the inputs coming into those output layers.
newOutputNames = [];
if ~isempty(lg.OutputNames)
    [tfs, locs] = ismember(lg.OutputNames, lg.Connections.Destination);
    assert(all(tfs))
    newOutputNames = lg.Connections.Source(locs);
    lg = lg.removeLayers( lg.OutputNames );
end
end

function warnings = iMaybeWarnaboutInputNorms(modifiedInputLayerNames)
warnings = message.empty;
if ~isempty(modifiedInputLayerNames)
    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterInputNormalization", ...
        join(modifiedInputLayerNames, ", "));
    warning(msg);
    warnings = msg;
end
end
