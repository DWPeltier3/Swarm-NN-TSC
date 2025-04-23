% Table of correspondence between MATLAB tensors and Tensorflow tensors:
%
% DLT format	                              TF format
% ----------                                  ---------
%
% Feature tensors:
% BC (dag network output only)                BC
% CB	                                      BC
% CU (when CT goes through globAvgPool1d 
%     or lstm('last'))	                      BC
%
% Spatial tensors:
% SC      (dlnet)	                          BSC
% SSC     (dlnet)	                          BSSC
% SSSC    (dlnet)	                          BSSSC
% SCB	                                      BSC
% SSCB	                                      BSSC
% SSSCB	                                      BSSSC
% SSSSCB	                                  BSSSSC     % For maxPooling2DLayer's "size" output.
%
% Temporal tensors:
% CT      (dlnet)	                          BTC
% CBT	                                      BTC
%
% Spatial+Temporal tensors:
% SCT     (dlnet)	                          BTSC
% SSCT    (dlnet)	                          BTSSC
% SSSCT   (dlnet)	                          BTSSSC
% SCBT	                                      BTSC
% SSCBT	                                      BTSSC
% SSSCBT	                                  BTSSSC
%
% U-only tensors (NOT SUPPORTED):
% UU      (dlnet)	                          BC
% UUU     (dlnet)	                          BSC
% UUUU    (dlnet)	                          BSSC

classdef LayerConverter

    %   Copyright 2022-2023 The MathWorks, Inc.

    % "toTensorflow" must be implemented by each subclass
    methods (Abstract)
        convertedLayer = toTensorflow(this, varargin);
    end

    % Useful properties for writing subclasses:
    properties (SetAccess=private)
        Layer              nnet.cnn.layer.Layer   % The external layer.
        NumInputs          double
        NumOutputs         double
        InputTensorName    string   % possibly an array
        OutputTensorName   string   % possibly an array
        InputFormat        string   % possibly an array
        OutputFormat       string   % possibly an array
        InputSize          cell
        OutputSize         cell

        % Internal properties
        networkAnalysis     nnet.internal.cnn.tf_exporter.NetworkAnalyzer
        layerNum(1,1)       {mustBeInteger}
        layerAnalyzer       nnet.internal.cnn.tf_exporter.LayerAnalyzer
        NameMap             containers.Map
        IsDanglingInput     logical
    end

    % Useful methods for writing subclasses:
    methods (Access=protected)
        % Method for generating a single line of Keras code
        function codeString = kerasCodeLine(this, InputTensorNames, OutputTensorNames, ...
                kerasFcnName, argsSprintfString, argsSprintfArgs, layerName, isTimeDistributed)
            % Makes a MIMO line of code calling a keras layer function,
            % optionally wrapping it in TimeDistributed, and optionally
            % named.
            arguments
                this                        % The object instance
                InputTensorNames    string  % Array of input names to the keras function.
                OutputTensorNames   string  % Array of output names of the keras function.
                kerasFcnName        string  % The Keras function name to be called.
                argsSprintfString   string  % A sprintf format string defining arguments to the Keras function. Omit the Name clause.
                argsSprintfArgs     cell    % Arguments to pass to the sprintf format string.
                layerName           string  % The name of the layer, or "" or empty if the layer has no weights (and thus needs no name).
                isTimeDistributed   logical % True if the Keras function should be wrapped in a call to TimeDistributed().
            end
            if isempty(layerName) || layerName==""
                nameStr = "";
            else
                nameStr = sprintf(", name=""%s""", layerName);
            end
            if isTimeDistributed
                % Put the name in the TimeDistributed, not in the layer.
                sprintfArgString = sprintf("%s = layers.TimeDistributed(%s(%s)%s)(%s)", ...
                    join(OutputTensorNames, ','), kerasFcnName, argsSprintfString, nameStr, join(InputTensorNames, ','));
            else
                % Put the name in the layer.
                sprintfArgString = sprintf("%s = %s(%s%s)(%s)", ...
                    join(OutputTensorNames, ','), kerasFcnName, argsSprintfString, nameStr, join(InputTensorNames, ','));
            end
            codeString = sprintf(sprintfArgString, argsSprintfArgs{:});
        end

        % Method to throw a warning without displaying a backtrace
        function warningNoBacktrace(~, msg)
            warnstate = warning('off','backtrace');
            C = onCleanup(@()warning(warnstate));
            warning(msg);
        end
    end


    % ONLY INTERNAL-USE METHODS ARE BELOW
    methods (Static)
        % Factory method for creating a LayerConverter of the right
        % subclass for the specified layer
        function subclassInstance = factory(networkAnalysis, layerNum, NameMap)
            assert(layerNum <= numel(networkAnalysis.LayerAnalyzers))
            layerAnalyzer = networkAnalysis.LayerAnalyzers(layerNum);

            % KEEP LAYERS IN ALPHABETICAL ORDER (ignoring package names).
            % Use comments on the right to specify the source of the layer
            % if it's not a DLT built-in layer.

            if isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.AdditionLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForAdditionLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.AveragePooling1DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForAveragePooling1DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.AveragePooling2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForAveragePooling2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.AveragePooling3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForAveragePooling3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.BatchNormalizationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForBatchNormalizationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.BiLSTMLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForBiLSTMLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.BinaryCrossEntropyRegressionLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForBinaryCrossEntropyRegressionLayer_Keras; % Keras/Tensorflow importer handwritten custom layer.
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.shufflenet.layer.ChannelShufflingLayer')              % From pretrained network
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForShufflenetChannelShufflingLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.ClassificationOutputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForClassificationOutputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.ClipLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForClipLayer_Keras;                       % Keras/Tensorflow importer handwritten custom layer.
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.ClippedReLULayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForClippedReLULayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.ConcatenationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForConcatenationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Convolution1DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForConvolution1DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Convolution2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForConvolution2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Convolution3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForConvolution3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Crop2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForCrop2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Crop3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForCrop3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.CrossChannelNormalizationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForCrossChannelNormalizationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.DepthConcatenationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForDepthConcatenationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.DepthToSpace2DLayer')                         % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForDepthToSpace2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.DicePixelClassificationLayer')              % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForDicePixelClassificationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.DropoutLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForDropoutLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.EmbeddingConcatenationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForEmbeddingConcatenationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.ELULayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForEluLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.FeatureInputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFeatureInputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.onnx.layer.Flatten3dInto2dLayer')                     % ONNX importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFlatten3dInto2dLayer_ONNX;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.onnx.layer.Flatten3dLayer')                           % ONNX importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFlatten3dLayer_ONNX;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.FlattenCStyleLayer') ...                  % Keras/Tensorflow importer handwritten custom layer.
                    || isa(layerAnalyzer.ExternalLayer, 'FlattenCStyleLayer')                                   % Audio toolbox 'openl3' pretrained network custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFlattenCStyleLayer_Keras;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.onnx.layer.FlattenInto2dLayer')                       % ONNX importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFlattenInto2dLayer_ONNX;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.FlattenLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFlattenLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.onnx.layer.FlattenLayer')                             % ONNX importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFlattenLayer_ONNX;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.FullyConnectedLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForFullyConnectedLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GELULayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGELULayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GlobalAveragePooling1DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGlobalAveragePooling1DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GlobalAveragePooling2DLayer') ...
                    || isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.GlobalAveragePooling2dLayer')         % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGlobalAveragePooling2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GlobalAveragePooling3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGlobalAveragePooling3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GlobalMaxPooling1DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGlobalMaxPooling1DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GlobalMaxPooling2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGlobalMaxPooling2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GlobalMaxPooling3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGlobalMaxPooling3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GroupedConvolution2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGroupedConvolution2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GroupNormalizationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGroupNormalizationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GRULayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGRULayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.GRUProjectedLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForGRUProjectedLayer;                
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Image3DInputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForImage3DInputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.ImageInputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForImageInputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.InputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForInputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.nasnetmobile.layer.NASNetMobileZeroPadding2dLayer') ||...
                    isa(layerAnalyzer.ExternalLayer, 'nnet.nasnetlarge.layer.NASNetLargeZeroPadding2dLayer')     % From pretrained network
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForNASNetMobileZeroPadding2dLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.InstanceNormalizationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForInstanceNormalizationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.LayerNormalizationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForLayerNormalizationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.LeakyReLULayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForLeakyReLULayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.LSTMLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForLSTMLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.LSTMProjectedLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForLSTMProjectedLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.MaxPooling1DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForMaxPooling1DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.MaxPooling2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForMaxPooling2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.MaxUnpooling2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForMaxUnpooling2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.MaxPooling3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForMaxPooling3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.MultiplicationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForMultiplicationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.PatchEmbeddingLayer')                       % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForPatchEmbeddingLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.PixelClassificationLayer')                  % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForPixelClassificationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.PointCloudInputLayer')                      % From Lidar Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForPointCloudInputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.PositionEmbeddingLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForPositionEmbeddingLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.PreluLayer')                              % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForPreluLayer_Keras;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.RegressionOutputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForRegressionOutputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.ReLULayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForReLULayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Resize2DLayer')                             % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForResize2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.Resize3DLayer')                             % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForResize3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.inceptionresnetv2.layer.ScalingFactorLayer')          % From pretrained network
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForInceptionresnetv2ScalingFactorLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SelfAttentionLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSelfAttentionLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SequenceFoldingLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSequenceFoldingLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SequenceInputLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSequenceInputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SequenceUnfoldingLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSequenceUnfoldingLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SigmoidLayer') ...
                    || isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.SigmoidLayer')                        % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSigmoidLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SoftmaxLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSoftmaxLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SpaceToDepthLayer')                         % From Computer Vision Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSpaceToDepthLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.SwishLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForSwishLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.TanhLayer') ...
                    || isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.TanhLayer')                           % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForTanhLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.TimeDistributedFlattenCStyleLayer')       % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForTimeDistributedFlattenCStyleLayer_Keras;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.TransposedConvolution1DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForTransposedConvolution1DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.TransposedConvolution2DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForTransposedConvolution2DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.TransposedConvolution3DLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForTransposedConvolution3DLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.WordEmbeddingLayer')                        % text Analytics Toolbox
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForWordEmbeddingLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.cnn.layer.YOLOv2OutputLayer')                         % From pretrained network
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForYOLOv2OutputLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.ZeroPadding1dLayer')                      % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForZeroPadding1dLayer_Keras;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.keras.layer.ZeroPadding2dLayer')                      % Keras/Tensorflow importer handwritten custom layer.
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForZeroPadding2dLayer_Keras;

                % Handle custom output layers that we don't know about:
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.layer.ClassificationLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForCustomClassificationLayer;
            elseif isa(layerAnalyzer.ExternalLayer, 'nnet.layer.RegressionLayer')
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForCustomRegressionLayer;

            else
                % Handle unsupported layers
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForUnsupportedLayer;
            end
            
            % Try converting the layer. If it fails for any reason, convert
            % it as an unsupported layer
            try
            subclassInstance = feval(constructor, networkAnalysis, layerNum, NameMap); %#ok<FVAL>
            catch
                constructor = @nnet.internal.cnn.tf_exporter.ConverterForUnsupportedLayer;
                subclassInstance = feval(constructor, networkAnalysis, layerNum, NameMap); %#ok<FVAL> 
            end
        end
    end

    methods
        % layerToTensorFlow is called by NetworkConverter to convert a
        % layer to Tensorflow. It calls the layer developer's toTensorflow
        % method, then calls checkLayer on it.
        function convertedLayer = layerToTensorflow(this, varargin)
            convertedLayer = toTensorflow(this, varargin{:});
            checkLayer(convertedLayer);
        end

        % A method to rename an input tensor. Currently used only by
        % NetworkConverter to handle dangling inputs.
        function this = renameInputTensor(this, inputNum, newName)
            this.InputTensorName(inputNum) = newName;
        end
    end

    methods (Access=protected)
        % Constructor
        function this = LayerConverter(networkAnalysis, layerNum, NameMap)
            % Set internal properties
            this.networkAnalysis = networkAnalysis;
            this.layerNum = layerNum;
            this.NameMap = NameMap;
            assert(layerNum <= numel(this.networkAnalysis.LayerAnalyzers))
            this.layerAnalyzer = this.networkAnalysis.LayerAnalyzers(layerNum);

            this.Layer = this.layerAnalyzer.ExternalLayer;
            this.NumInputs = this.layerAnalyzer.NumInputs;
            this.NumOutputs = this.layerAnalyzer.NumOutputs;

            % Set input tensor properties
            for i=1:this.NumInputs
                rawSourceName = inputSourceName(this.layerAnalyzer, i);
                if startsWith(rawSourceName, "/")
                    % This is a dangling network input
                    this.IsDanglingInput(i) = true;
                    inputNumber = str2double(rawSourceName.extractAfter(1));
                    assert(~isempty(inputNumber))
                    netInputName = this.NameMap(networkAnalysis.InputNames(inputNumber));
                    this.InputTensorName(i) = netInputName;
                else
                    this.IsDanglingInput(i) = false;
                    this.InputTensorName(i) = string(this.NameMap(char(inputSourceName(this.layerAnalyzer, i))));
                end
                this.InputFormat(i) = inputFormat(this.layerAnalyzer, i);
                this.InputSize{i} = inputSize(this.layerAnalyzer, i);
            end

            % Set output tensor properties
            if this.layerAnalyzer.IsInputLayer || this.layerAnalyzer.IsOutputLayer
                this.OutputTensorName(1) = string(this.NameMap(this.Layer.Name));
                this.OutputFormat(1) = outputFormat(this.layerAnalyzer, 1);
                this.OutputSize{1} = outputSize(this.layerAnalyzer, 1);
            else
                for i=1:this.NumOutputs
                    this.OutputTensorName(i) = string(this.NameMap(char(this.layerAnalyzer.OutputNames(i))));
                    this.OutputFormat(i) = outputFormat(this.layerAnalyzer, i);
                    this.OutputSize{i} = outputSize(this.layerAnalyzer, i);
                end
            end
        end
    end
end