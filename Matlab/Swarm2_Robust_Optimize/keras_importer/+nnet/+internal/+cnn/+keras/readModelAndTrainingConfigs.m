function [ModelConfig, TrainingConfig, KerasVersion] = readModelAndTrainingConfigs(ConfigFile, Format)

% Copyright 2017-2019 The Mathworks, Inc.

TrainingConfig = [];
switch Format
    case 'hdf5'
        iVerifyKerasBackend(ConfigFile);
        iVerifyH5ModelConfigAttribute(ConfigFile);
        try
            ModelConfigAtt = h5readatt(ConfigFile, '/', 'model_config');
            ModelConfig = jsondecode(ModelConfigAtt);
            Info = h5info(ConfigFile);
            if ~ismember('keras_version', {Info.Attributes.Name})
                iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:NoKerasVersion', ConfigFile);
            else
                KerasVersion = h5readatt(ConfigFile, '/', 'keras_version');
            end
            iVerifyKerasVersion(ConfigFile, KerasVersion);
        catch ME
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantReadModelConfig', ConfigFile, ME.message)));
        end
        if iH5HasTrainingConfigAttribute(ConfigFile)
            try
                TrainingConfigAtt = h5readatt(ConfigFile, '/', 'training_config');
                TrainingConfig = jsondecode(TrainingConfigAtt);
            catch ME
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantReadTrainingConfig', ConfigFile, ME.message)));
            end
        end
    case 'json'
        try
            ModelConfig = jsondecode(fileread(ConfigFile));
            KerasVersion = ModelConfig.keras_version;
            iVerifyKerasVersion(ConfigFile, KerasVersion);
        catch ME
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantReadModelConfig', ConfigFile, ME.message)));
        end
    otherwise
        assert(false);
end
end

function iVerifyKerasVersion(ConfigFile, KerasVersion)

OLDEST_SUPPORTED_KERAS_VERSION = '2.0.0';
NEWEST_SUPPORTED_KERAS_VERSION = '2.6.0';

import nnet.internal.cnn.keras.*
if ver2num(KerasVersion) < ver2num(OLDEST_SUPPORTED_KERAS_VERSION)
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:KerasVersionTooOld', ...
        ConfigFile, KerasVersion, OLDEST_SUPPORTED_KERAS_VERSION)));
end
if ver2num(KerasVersion) > ver2num(NEWEST_SUPPORTED_KERAS_VERSION)
    iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:KerasVersionTooNew', ...
        ConfigFile, KerasVersion, NEWEST_SUPPORTED_KERAS_VERSION);
end
end

function iVerifyKerasBackend(ConfigFile)
try
    Info = h5info(ConfigFile);
catch ME
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantReadHDF5', ConfigFile, ME.message)));
end
if ~ismember('backend', {Info.Attributes.Name})
    iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnknownBackend', ConfigFile);
else
    Backend = h5readatt(ConfigFile, '/', 'backend');
    if ~isequal(Backend, 'tensorflow')
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedBackend', ConfigFile, Backend)));
    end
end
end

function iVerifyH5ModelConfigAttribute(ConfigFile)
% Assume file exists
try
    Info = h5info(ConfigFile);
catch ME
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:CantReadHDF5', ConfigFile, ME.message)));
end
if ~ismember('model_config', {Info.Attributes.Name})
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NoModelConfig', ConfigFile)));
end
end

function tf = iH5HasTrainingConfigAttribute(ConfigFile)
Info = h5info(ConfigFile);
tf = ismember('training_config', {Info.Attributes.Name});
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end