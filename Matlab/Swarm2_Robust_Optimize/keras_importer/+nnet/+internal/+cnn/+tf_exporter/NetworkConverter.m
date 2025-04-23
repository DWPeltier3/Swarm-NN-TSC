classdef NetworkConverter

    %   Copyright 2022-2023 The MathWorks, Inc.

    properties (SetAccess=private)
        net     % Can be a SeriesNetwork, DAGNetwork, dlnetwork, LayerGraph
    end

    methods
        function this = NetworkConverter(net)
            this.net = net;
        end

        function convertedNetwork = networkToTensorflow(this, initialWarnings)
            % Make a projected network exportable.
            this.net = deep.internal.sdk.projection.prepareProjectedNetworkFor3pExporters(this.net);

            % DLT net analysis and renaming
            networkAnalyzer = nnet.internal.cnn.tf_exporter.NetworkAnalyzer(this.net);
            NameMap         = iMakeNameMap(networkAnalyzer);
            % network conversion
            netInputNames = arrayfun(@(n)string(NameMap(n)), networkAnalyzer.InputNames);
            netOutputNames = arrayfun(@(n)string(NameMap(n)), networkAnalyzer.OutputNames);
            convertedNetwork = nnet.internal.cnn.tf_exporter.ConvertedNetwork(netInputNames, netOutputNames, initialWarnings);
            % Check for quantized net
            if (isa(this.net, "SeriesNetwork") || isa(this.net, "DAGNetwork") || isa(this.net, "dlnetwork")) ...
                    && quantizationDetails(this.net).IsQuantized
                % Update warnings
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterQuantizedNet");
                iWarningNoBacktrace(msg);
                convertedNetwork = addWarningMessages(convertedNetwork, msg); 
            end

            layers = [networkAnalyzer.LayerAnalyzers.ExternalLayer];
            % (1) Create all layer converters
            for layerNum = 1:numel(layers)
                layerConverters{layerNum} = nnet.internal.cnn.tf_exporter.LayerConverter.factory(networkAnalyzer, layerNum, NameMap); %#ok<AGROW>
            end
            % (2) Convert all input layers and dangling input tensors
            % before doing any other layers
            for layerNum = 1:numel(layers)
                if layerConverters{layerNum}.layerAnalyzer.IsInputLayer
                    % Convert this input layer
                    convertedLayer   = layerToTensorflow(layerConverters{layerNum});
                    convertedNetwork = updateFromConvertedLayer(convertedNetwork, convertedLayer);
                else
                    % Convert any dangling input tensors coming into this layer
                    for inputNum = 1:layerConverters{layerNum}.NumInputs
                        if layerConverters{layerNum}.IsDanglingInput(inputNum)
                            converter = nnet.internal.cnn.tf_exporter.ConverterForDanglingInputTensor(networkAnalyzer, layerNum, NameMap);
                            convertedLayer = layerToTensorflow(converter, inputNum);
                            convertedNetwork = updateFromConvertedLayer(convertedNetwork, convertedLayer);
                            % ConverterForDanglingInputTensor will have
                            % renamed the input tensor by adding the suffix
                            % "_input". So modify layerConverters{layerNum}
                            % to report the new name as its input tensor
                            % name:
                            newName = layerConverters{layerNum}.InputTensorName(inputNum) + "_input";
                            layerConverters{layerNum} = renameInputTensor(layerConverters{layerNum}, inputNum, newName);
                        end
                    end
                end
            end
            % (3) Convert all non-input layers
            for layerNum = 1:numel(layers)
                converter = layerConverters{layerNum};
                if ~converter.layerAnalyzer.IsInputLayer
                    convertedLayer = layerToTensorflow(converter);
                    % If conversion failed, collect warnings, then treat it
                    % as an unsupported layer
                    if ~convertedLayer.Success
                        convertedNetwork.WarningMessages = [convertedNetwork.WarningMessages, convertedLayer.WarningMessages];
                        newConverter = nnet.internal.cnn.tf_exporter.ConverterForUnsupportedLayer(...
                            converter.networkAnalysis, converter.layerNum, converter.NameMap);
                        convertedLayer = layerToTensorflow(newConverter);
                    end
                    convertedNetwork = updateFromConvertedLayer(convertedNetwork, convertedLayer);
                end
            end
        end
    end
end

function NameMap = iMakeNameMap(networkAnalyzer)
% The NameMap takes a DLT output connection port as input (e.g., 'layer0',
% 'layer1/out') and outputs the name of the TF tensor representing that
% tensor in the generated code.
layers = [networkAnalyzer.LayerAnalyzers.ExternalLayer];
uniqueNames = string.empty;
% Add tensors created by layers:
for i=1:numel(layers)
    layerAnalyzer = networkAnalyzer.LayerAnalyzers(i);
    layerName = string(layers(i).Name);
    outputNames = string(layerAnalyzer.OutputNames);
    if numel(outputNames)==1 || layerAnalyzer.IsOutputLayer
        % The output of a single-output layer can be referred to by the
        % layerName alone.
        % NOTE: Output layers have an empty 'OutputNames' property! The
        % layer's name is its output name.
        uniqueNames = [uniqueNames, layerName];
    else
        uniqueNames = [uniqueNames, outputNames];
    end
end
% Add dangling input names
uniqueNames = unique([uniqueNames, string(networkAnalyzer.InputNames)]);
% Make the initial NameMap
NameMap = iMakeNamesPythonCompatible(uniqueNames);
% Now add extra "/<name>" entries for layers with only one output
for i=1:numel(layers)
    layerAnalyzer = networkAnalyzer.LayerAnalyzers(i);
    layerName = string(layers(i).Name);
    outputNames = string(layerAnalyzer.OutputNames);
    if numel(outputNames)==1
        tensorName = NameMap(layerName);
        NameMap(outputNames) = tensorName;
    end
end
end

function NameMap = iMakeNamesPythonCompatible(UniqueNames)
UniquePythonNames   = iRenamePythonKeywords(UniqueNames);
MATLABNames         = matlab.lang.makeValidName(UniquePythonNames);
UniqueMATLABNames   = matlab.lang.makeUniqueStrings(MATLABNames, {}, 50);   % 63 is the MATLAB limit
NameMap             = containers.Map(UniqueNames, UniqueMATLABNames);
NameMap('')         = 'unknownInput';
end

function UniqueNames = iRenamePythonKeywords(UniqueNames)
keywords = ["as","def","else","for","from","if","import","in","lambda","return","with"];
isKeyword = ismember(UniqueNames, keywords);
UniqueNames(isKeyword) = UniqueNames(isKeyword) + "_x";
end

function iWarningNoBacktrace(msg)
warnstate = warning('off','backtrace');
C = onCleanup(@()warning(warnstate));
warning(msg);
end
