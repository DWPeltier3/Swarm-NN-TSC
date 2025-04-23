function LayersOrGraph = importKerasLayers(ConfigFile, varargin)

% Copyright 2017-2023 The MathWorks, Inc.
% Register message catalog
nnet.internal.cnn.keras.setAdditionalResourceLocation();

% Warn about importTensorFlowLayer deprecation
nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WarnAPIDeprecationKeras', 'importKerasLayers');

% Check input
[ConfigFile, Format, ImportWeights, WeightFile, OutputLayerType, UserImageInputSize] = iValidateInputs(ConfigFile, varargin{:});
% Read ModelConfig, and TrainingConfig if present
[ModelConfig, TrainingConfig, ~] = nnet.internal.cnn.keras.readModelAndTrainingConfigs(ConfigFile, Format);
% Verify that TrainingConfig exists so we can find the loss function
TrainingConfig = iUpdateTrainingConfig(TrainingConfig, ModelConfig, OutputLayerType);
% Parse model config
KM = nnet.internal.cnn.keras.ParsedKerasModel(ModelConfig, TrainingConfig);
% Assemble model and import weights
if ImportWeights
    AM = nnet.internal.cnn.keras.AssembledModel(KM, WeightFile);
else
    AM = nnet.internal.cnn.keras.AssembledModel(KM);
end
nnet.internal.cnn.keras.util.checkInputSizeForMINetworks(AM, UserImageInputSize, false); 
% Translate model
LayersOrGraph = translateAssembledModel(AM, TrainingConfig, ImportWeights, false, UserImageInputSize);
% Warn if there are unsupported layers
[placeholderLayers, placeholderindices] = findPlaceholderLayers(LayersOrGraph);
if ~isempty(placeholderLayers)
    iNotifyPlaceholderLayers(placeholderLayers, numel(AM.InputLayerIndices));
end
[layerGraph, minLengthRequired] = nnet.internal.cnn.keras.util.checkMinLengthRequired(LayersOrGraph);
if minLengthRequired
    LayersOrGraph = nnet.internal.cnn.keras.util.autoSetMinLength(layerGraph);
end
LayersOrGraph = nnet.internal.cnn.keras.util.configureOutputLayer(LayersOrGraph, OutputLayerType, placeholderindices); 
LayersOrGraph = LayersOrGraph(:);
end

%% Create a TrainingConfig 
function TrainingConfig = iUpdateTrainingConfig(TrainingConfig, ModelConfig, OutputLayerType)
if isempty(TrainingConfig) || isempty(TrainingConfig.loss)
    numOutputs = 1; 
    if isfield(ModelConfig.config, 'output_layers')
        numOutputs = numel(ModelConfig.config.output_layers);
    end 
    if ~isempty(OutputLayerType) && numOutputs > 1
        % MO network and OutputLayerType are not compatible
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MOOutputLayerType')));
    elseif numOutputs > 1
        % Dummy TrainingConfig with 'None' as all output's loss. This will
        % result in a placeholder output layer. 
        TrainingConfig = struct;
        for i = 1:numOutputs
            curOutputName = ModelConfig.config.output_layers{i}{1}; 
            TrainingConfig.loss.(curOutputName) = 'None'; 
        end
    elseif isempty(OutputLayerType)
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:LayersNoTrainingConfig');
    else        
        TrainingConfig = nnet.internal.cnn.keras.util.getTrainingConfig(OutputLayerType);
    end
else
    % There's a training config already
    if ~isempty(OutputLayerType) && ~strcmpi(OutputLayerType, 'PixelClassification')
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:OutputLayerTypeNotNeeded');
    end
end
end

%% Input validation
function [ConfigFile, ConfigFileFormat, ImportWeights, WeightFile, OutputLayerType, ImageInputSize] = iValidateInputs(ConfigFile, varargin)
par = inputParser();
par.addRequired('ConfigFile');
par.addParameter('ImportWeights', false);
par.addParameter('WeightFile', '');
par.addParameter('OutputLayerType', '');
par.addParameter('ImageInputSize', []);
par.parse(ConfigFile,varargin{:});
ImportWeights = par.Results.ImportWeights;
WeightFile = par.Results.WeightFile;
OutputLayerType = par.Results.OutputLayerType;
ImageInputSize = par.Results.ImageInputSize;
ConfigFile = nnet.internal.cnn.keras.util.validateConfigFile(ConfigFile, false);
ConfigFileFormat = nnet.internal.cnn.keras.util.getFileFormat(ConfigFile, false);
iValidateImportWeights(ImportWeights);
if isempty(WeightFile)
    if ImportWeights && isequal(ConfigFileFormat, 'json')
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantImportWeightsFromJsonFile', ConfigFile)));
    else
        WeightFile = ConfigFile;
    end
end
WeightFile = nnet.internal.cnn.keras.util.validateKerasWeightFile(WeightFile);
OutputLayerType = nnet.internal.cnn.keras.util.validateKerasOutputLayerType(OutputLayerType);
ImageInputSize = nnet.internal.cnn.keras.util.validateImageInputSize(ImageInputSize);
end

function iValidateImportWeights(ImportWeights)
if ~(isscalar(ImportWeights) && islogical(ImportWeights) || isequal(ImportWeights,0) || isequal(ImportWeights,1))
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:BadImportWeights')));
end
end

function iNotifyPlaceholderLayers(placeholderLayers, numInputs)
if any(arrayfun(@(l)(isa(l, 'nnet.keras.layer.PlaceholderLayer')), placeholderLayers))
    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedLayerWarning');
end

if any(arrayfun(@(l)(isa(l, 'nnet.keras.layer.PlaceholderInputLayer')), placeholderLayers))
    if numInputs == 1
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:LayersImageInputSizeNeeded');
    else 
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:MILayersImageInputSizeNeeded');
    end 
end

if any(arrayfun(@(l)(isa(l, 'nnet.keras.layer.PlaceholderOutputLayer')), placeholderLayers))
    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedOutputLayerWarning');
end
end
