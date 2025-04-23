function Network = importKerasNetwork(ConfigFile, varargin)

% Copyright 2017-2023 The MathWorks, Inc.
% Register message catalog
nnet.internal.cnn.keras.setAdditionalResourceLocation();

% Warn about importTensorFlowLayer deprecation
nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WarnAPIDeprecationKeras', 'importKerasNetwork');

% Check input
[ConfigFile, Format, WeightFile, OutputLayerType, UserImageInputSize, Classes] = iValidateInputs(ConfigFile, varargin{:});
if isempty(WeightFile)
    switch Format
        case 'hdf5'
            WeightFile = ConfigFile;
        case 'json'
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ConfigJSONButNoWeightfile')));
        otherwise
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedFormat', ext)));
    end
end
% Read ModelConfig, and TrainingConfig if present
[ModelConfig, TrainingConfig, ~] = nnet.internal.cnn.keras.readModelAndTrainingConfigs(ConfigFile, Format);
TrainingConfig = iUpdateTrainingConfig(TrainingConfig, ModelConfig, OutputLayerType);
% Parse model config
KM = nnet.internal.cnn.keras.ParsedKerasModel(ModelConfig, TrainingConfig);
% Assemble model and import weights
AM = nnet.internal.cnn.keras.AssembledModel(KM, WeightFile);
nnet.internal.cnn.keras.util.checkInputSizeForMINetworks(AM, UserImageInputSize, true); 
% Translate model into layers
ImportWeights   = true;
TrainOptionsRequested = false;
LayersOrGraph   = translateAssembledModel(AM, TrainingConfig, ImportWeights, TrainOptionsRequested, UserImageInputSize);
placeholderLayers = findPlaceholderLayers(LayersOrGraph);
iNotifyImageInputSizeNeeded(placeholderLayers);
[layerGraph, minLengthRequired] = nnet.internal.cnn.keras.util.checkMinLengthRequired(LayersOrGraph);
if minLengthRequired
    LayersOrGraph = nnet.internal.cnn.keras.util.autoSetMinLength(layerGraph);
end
LayersOrGraph      = nnet.internal.cnn.keras.util.configureOutputLayerForNetwork(LayersOrGraph, Classes);
C               = iSuppressAutoClassesWarning();
Network         = assembleNetwork(LayersOrGraph);
end

%% Create a TrainingConfig
function TrainingConfig = iUpdateTrainingConfig(TrainingConfig, ModelConfig, OutputLayerType)
if isempty(TrainingConfig) || isempty(TrainingConfig.loss)
    numOutputs = 1; 
    if isfield(ModelConfig.config, 'output_layers')
        numOutputs = numel(ModelConfig.config.output_layers); 
    end    
    if numOutputs > 1 
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MONetworkNoTrainingConfig'))); 
    elseif isempty(OutputLayerType)
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NetworkNoTrainingConfig')));
    else
        TrainingConfig = nnet.internal.cnn.keras.util.getTrainingConfig(OutputLayerType);
    end
else
    % There's a training config already
    if ~isempty(OutputLayerType)
         nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:OutputLayerTypeNotNeeded');
    end
end
end

%% Input validation
function [ConfigFile, Format, WeightFile, OutputLayerType, ...
    ImageInputSize, Classes] = iValidateInputs(ConfigFile, varargin)
defaultClasses = 'auto';
defaultClassNames = {};
par = inputParser();
par.addRequired('ConfigFile');
par.addParameter('WeightFile', '');
par.addParameter('OutputLayerType', '');
par.addParameter('ImageInputSize', []);
par.addParameter('ClassNames', defaultClassNames);
par.addParameter('Classes', defaultClasses, @iAssertValidClasses);

par.parse(ConfigFile,varargin{:});
WeightFile = par.Results.WeightFile;
OutputLayerType = par.Results.OutputLayerType;
ImageInputSize = par.Results.ImageInputSize;
ClassNames = par.Results.ClassNames;
Classes = par.Results.Classes;

ConfigFile = nnet.internal.cnn.keras.util.validateConfigFile(ConfigFile, true);
Format = nnet.internal.cnn.keras.util.getFileFormat(ConfigFile, true);
WeightFile = nnet.internal.cnn.keras.util.validateKerasWeightFile(WeightFile);
OutputLayerType = nnet.internal.cnn.keras.util.validateKerasOutputLayerType(OutputLayerType);
ImageInputSize = nnet.internal.cnn.keras.util.validateImageInputSize(ImageInputSize);

if iIsSpecified(ClassNames, defaultClassNames) && ...
        iIsSpecified(Classes, defaultClasses)
    throwAsCaller(MException(message(...
        'nnet_cnn_kerasimporter:keras_importer:ClassesAndClassNamesNVP')))
elseif iIsSpecified(ClassNames, defaultClassNames)
    warning(message('nnet_cnn_kerasimporter:keras_importer:ClassNamesDeprecated')); 
    ClassNames = iValidateClassNames(ClassNames);
    Classes = categorical(ClassNames, ClassNames);
elseif iIsSpecified(Classes, defaultClasses)
    Classes = iConvertClassesToCanonicalForm(Classes);
else
    % Not specified ClassNames nor Classes. Do nothing.
end
end

function tf = iIsSpecified(value, defaultValue)
tf = ~isequal(convertStringsToChars(value), defaultValue);
end

function  ClassNames = iValidateClassNames(ClassNames)
if isstring(ClassNames)
    ClassNames = cellstr(ClassNames);
elseif ~iscellstr(ClassNames)
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:InvalidClassNames')));
end
if ~isvector(ClassNames)
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:InvalidClassNames')));
end
% make sure it's a column vector
ClassNames = ClassNames(:);
end

function iAssertValidClasses(value)
nnet.internal.cnn.layer.paramvalidation.validateClasses(value);
end

function classes = iConvertClassesToCanonicalForm(classes)
classes = ...
    nnet.internal.cnn.layer.paramvalidation.convertClassesToCanonicalForm(classes);
end

function C = iSuppressAutoClassesWarning()
warnState = warning('off', 'nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasWarnings');
C = onCleanup(@()warning(warnState));
end

function iNotifyImageInputSizeNeeded(placeholderLayers) 
if any(arrayfun(@(l)(isa(l, 'nnet.keras.layer.PlaceholderInputLayer')), placeholderLayers))
    % For single input networks, if the image input size is unknown, then
    % tell the user to use ImageInputSize argument. 
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ImageInputSizeNeeded')));
end
end
