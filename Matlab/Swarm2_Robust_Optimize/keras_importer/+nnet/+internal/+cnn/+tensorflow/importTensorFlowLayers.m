function [layerGraph] = importTensorFlowLayers(path, options)
%

% Copyright 2021-2023 The MathWorks, Inc.
    arguments 
        path {mustBeFolder}
        options.PackageName {mustBeTextScalar} = ''
        options.Namespace {mustBeTextScalar} = ''
        options.OutputLayerType (1, :) char {mustBeMember(options.OutputLayerType, {'classification', 'regression', 'pixelclassification'})}
        options.ImageInputSize (1, :) {mustBeNumeric} = []
        options.TargetNetwork (1, :) {mustBeMember(options.TargetNetwork, {'dlnetwork', 'dagnetwork'})} = 'dagnetwork'
        options.Verbose (1,:) {mustBeInRange(options.Verbose, 0,1)} = 1
    end 
    import nnet.internal.cnn.tensorflow.*;
    importManager = nnet.internal.cnn.tensorflow.ImportManager(path, options);

    % Warn about importTensorFlowLayer deprecation
    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WarnAPIDeprecation', 'importTensorFlowLayers');
    
    % Warn about PackageName argument deprecation
    if(~isempty(options.PackageName))
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WarnArgumentDeprecation');
    end

    % Copy Namespace value to PackageName if Namespace is not empty
    if(~isempty(options.Namespace))
        options.PackageName = options.Namespace;
    end

    options.PackageName = char(options.PackageName);
	path = nnet.internal.cnn.tensorflow.util.validateTFModelFolder(path);									  
	nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseImportStarts', options.Verbose);
    options.ImageInputSize = nnet.internal.cnn.keras.util.validateImageInputSize(options.ImageInputSize);
    sm = savedmodel.TFSavedModel(path, importManager, false); 
    
    % parse inputs 
    if strcmp(options.TargetNetwork, 'dlnetwork') 
        if nnet.internal.cnn.tensorflow.util.hasFoldingLayer(sm.KerasManager.LayerGraph)
            nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:SequenceFoldingNotCompatibleWithDlnetwork');
        elseif isfield(options,'OutputLayerType')
            nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:OutputLayerTypeNotNeededForDlnetwork');
        end
    end
    
    nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseTranslationStarts', options.Verbose);
    if isempty(options.PackageName)
       [layerGraph, hasUnsupportedOp] = gcl.translateTFKerasLayers(sm); 
    else 
       [layerGraph, hasUnsupportedOp] = gcl.translateTFKerasLayers(sm, options.PackageName); 
    end
        
    [~, placeholderindices] = findPlaceholderLayers(layerGraph);
    [layerGraph, minLengthRequired] = nnet.internal.cnn.keras.util.checkMinLengthRequired(layerGraph);
    if minLengthRequired
        layerGraph = nnet.internal.cnn.keras.util.autoSetMinLength(layerGraph);
    end
    if isfield(options,'OutputLayerType')
        layerGraph = nnet.internal.cnn.keras.util.configureOutputLayer(layerGraph, options.OutputLayerType, placeholderindices);
    end
    
    if hasUnsupportedOp
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:GeneratedLayerContainsUnsupportedOpWarning');
    end
    
    nnet.internal.cnn.tensorflow.util.displayVerboseMessage('nnet_cnn_kerasimporter:keras_importer:VerboseTranslationFinished', options.Verbose);
end 
