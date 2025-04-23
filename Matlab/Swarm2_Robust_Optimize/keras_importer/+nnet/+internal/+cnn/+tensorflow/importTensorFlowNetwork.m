function [net] = importTensorFlowNetwork(path, options)
%

% Copyright 2021-2023 The MathWorks, Inc.
    arguments 
        path {mustBeFolder}
        options.PackageName {mustBeTextScalar} = ''
        options.Namespace {mustBeTextScalar} = ''
        options.OutputLayerType (1, :) char {mustBeMember(options.OutputLayerType, {'classification', 'regression', 'pixelclassification'})}
        options.ImageInputSize (1, :) {mustBeNumeric} = []
        options.TargetNetwork (1, :) {mustBeMember(options.TargetNetwork, {'dlnetwork', 'dagnetwork'})} = 'dagnetwork'
        options.Classes (1, :) {nnet.internal.cnn.layer.paramvalidation.validateClasses(options.Classes)} = 'auto'
        options.Verbose (1, :) {mustBeInRange(options.Verbose, 0,1)} = 1
        options.OnlySupportDlnetwork (1, :) {mustBeNumericOrLogical} = 0
    end
    % Warn about importTensorFlowNetwork deprecation
    if ~options.OnlySupportDlnetwork
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WarnAPIDeprecation', 'importTensorFlowNetwork');
    end

    % Copy Namespace value to PackageName if Namespace is not empty
    if(~isempty(options.PackageName))
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WarnArgumentDeprecation');
    end
    if(~isempty(options.Namespace))
        options.PackageName = options.Namespace;
    end
    
    importManager = nnet.internal.cnn.tensorflow.ImportManager(path, options);
    net = importManager.translate;
end