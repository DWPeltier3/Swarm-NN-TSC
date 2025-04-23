classdef LayerTranslator
    % To add a new kind of Keras layer to be translated, subclass this
    % class, define the abstract property and the abstract method below,
    % and add a case to the 'create' factory method.

% Copyright 2017-2023 The MathWorks, Inc.

    
    properties(Abstract, Constant)
        % A cellstr of the names of keras weight parameters, in any order.
        % E.g., for Conv2D the value is {'kernel', 'bias'}. For LSTM the
        % value is {'kernel', 'bias', 'recurrent_kernel'}.
        WeightNames
    end
    
    methods(Abstract)
        % translate() Takes a LayerSpec for the layer and whether to
        % translate the weights and training params, and outputs a cell
        % array of NNT layers. Do not append output layers -- those are
        % handled automatically.
        NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize);
    end
    
    methods % Overrideable
        % canSupportSettings() should return true if and only if the
        % specific settings in the KerasConfig are supportable. Override
        % this if there are any known unsupportable settings.
        function [tf, Message] = canSupportSettings(this, LSpec)
            tf = true;
            Message = '';
        end
    end
    
    properties(SetAccess=protected)
        KerasLayerType
    end
    
    methods(Static)
        function Obj = create(KerasLayerType, LSpec, isDAGNetwork)
            % Create and return an instance of a subclass corresponding to
            % the KerasLayerType. If the layer is unknown or has
            % unsupported settings, return a placeholder layer translator.
            if nargin < 3
                isTFLayer = false;
            end

            switch KerasLayerType
                % Deep Learning Toolbox (DLT) layers:
                case 'Activation'
                    Obj = nnet.internal.cnn.keras.TranslatorForActivationLayer;
                case 'Add'
                    Obj = nnet.internal.cnn.keras.TranslatorForAddLayer;
                case 'AveragePooling1D'
                    Obj = nnet.internal.cnn.keras.TranslatorForAveragePooling1DLayer;
                case 'AveragePooling2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForAveragePooling2DLayer;
                case 'AveragePooling3D'
                    Obj = nnet.internal.cnn.keras.TranslatorForAveragePooling3DLayer;
                case 'BatchNormalization'
                    Obj = nnet.internal.cnn.keras.TranslatorForBatchNormalizationLayer;
                case 'Bidirectional'
                    if isBidirectionalLayer(LSpec)
                        if isequal(LSpec.KerasConfig.layer.class_name, 'LSTM')
                            Obj = nnet.internal.cnn.keras.TranslatorForBidirectionalLSTMLayer;
                        elseif isequal(LSpec.KerasConfig.layer.class_name, 'CuDNNLSTM')
                            Obj = nnet.internal.cnn.keras.TranslatorForBidirectionalCuDNNLSTMLayer;
                        else
                            Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                        end
                    else
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                case 'Concatenate'
                    if ~LSpec.isTensorFlowLayer
                        % Call from Keras importer, keep old behavior
                        Obj = nnet.internal.cnn.keras.TranslatorForConcatenateLayer;
                    else
                        % Call from TF importer, create a placeholder layer
                        % which will be replaced by a GCL
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                case 'Conv1D'
                    Obj = nnet.internal.cnn.keras.TranslatorForConv1DLayer;
                case 'Conv2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForConv2DLayer;
                case 'Conv3D'
                    Obj = nnet.internal.cnn.keras.TranslatorForConv3DLayer;
                case 'Conv2DTranspose'
                    Obj = nnet.internal.cnn.keras.TranslatorForConv2DTransposeLayer;
                case 'Conv3DTranspose'
                    Obj = nnet.internal.cnn.keras.TranslatorForConv3DTransposeLayer;
                case 'Dense'
                    if ~LSpec.isTensorFlowLayer
                        % Call from Keras importer, keep old behavior
                        Obj = nnet.internal.cnn.keras.TranslatorForDenseLayer;
                    else
                        % Call from TF importer, create a placeholder layer
                        % which will be replaced by a GCL
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                case 'DepthwiseConv2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForDepthwiseConv2DLayer;                    
                case 'Dropout'
                    Obj = nnet.internal.cnn.keras.TranslatorForDropoutLayer;
                case 'Flatten'
                    if ~LSpec.IsTimeDistributed
                        Obj = nnet.internal.cnn.keras.TranslatorForFlattenLayer;                        
                    else
                        Obj = nnet.internal.cnn.keras.TranslatorForTimeDistributedFlattenLayer;
                    end
                case 'GlobalAveragePooling1D'
                    Obj = nnet.internal.cnn.keras.TranslatorForGlobalAveragePooling1DLayer;
                case 'GlobalAveragePooling2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForGlobalAveragePooling2DLayer;
                case 'GlobalAveragePooling3D'
                    Obj = nnet.internal.cnn.keras.TranslatorForGlobalAveragePooling3DLayer;
                case 'GlobalMaxPooling1D'
                    Obj = nnet.internal.cnn.keras.TranslatorForGlobalMaxPooling1DLayer;
                case 'GlobalMaxPooling2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForGlobalMaxPooling2DLayer;
                case 'GlobalMaxPooling3D'
                    Obj = nnet.internal.cnn.keras.TranslatorForGlobalMaxPooling3DLayer;
                case 'GRU'
                    Obj = nnet.internal.cnn.keras.TranslatorForGRULayer;
                case 'CuDNNGRU'
                    Obj = nnet.internal.cnn.keras.TranslatorForCuDNNGRULayer;
                case 'InputLayer'
                    if ~LSpec.isTensorFlowLayer
                        % Call from Keras importer, keep old behavior
                        Obj = nnet.internal.cnn.keras.TranslatorForDAGInputLayer;
                    else
                        % Call from TF importer
                        Obj = nnet.internal.cnn.keras.TranslatorForInputLayers;
                    end
                case 'LeakyReLU'
                    Obj = nnet.internal.cnn.keras.TranslatorForLeakyReLULayer;
                case 'ELU'
                    Obj = nnet.internal.cnn.keras.TranslatorForELULayer;
                case 'LSTM'
                    Obj = nnet.internal.cnn.keras.TranslatorForLSTMLayer;
                case 'CuDNNLSTM'
                    Obj = nnet.internal.cnn.keras.TranslatorForCuDNNLSTMLayer;
                case 'MaxPooling1D'
                    Obj = nnet.internal.cnn.keras.TranslatorForMaxPooling1DLayer;
                case 'MaxPooling2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForMaxPooling2DLayer;
                case 'MaxPooling3D'
                    Obj = nnet.internal.cnn.keras.TranslatorForMaxPooling3DLayer;
                case 'Merge'
                    Obj = nnet.internal.cnn.keras.TranslatorForMergeLayer;
                case 'Multiply'
                    Obj = nnet.internal.cnn.keras.TranslatorForMultiplyLayer;
                case 'PReLU'
                    if ~LSpec.isTensorFlowLayer || isDAGNetwork
                        % Preserve Functionality for Keras Importer and TF
                        % DAGNetwork import
                        Obj = nnet.internal.cnn.keras.TranslatorForPReLULayerLegacy; 
                    else
                        Obj = nnet.internal.cnn.keras.TranslatorForPReLULayer;
                    end
                case 'ReLU'
                    Obj = nnet.internal.cnn.keras.TranslatorForReLULayer;
                case 'Reshape'
                    if ~LSpec.isTensorFlowLayer
                        % Call from Keras importer, keep old behavior
                        Obj = nnet.internal.cnn.keras.TranslatorForReshapeLayer;
                    else
                        % Call from TF importer, create a placeholder layer
                        % which will be replaced by a GCL
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                case 'SeparableConv2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForSeparableConv2DLayer; 
                case 'Softmax'
                    Obj = nnet.internal.cnn.keras.TranslatorForSoftmaxLayer;
                case 'TFOpLambda'
                    Obj = nnet.internal.cnn.keras.TranslatorForTFOpLambdaLayer;
                case 'ZeroPadding1D'
                    Obj = nnet.internal.cnn.keras.TranslatorForZeroPadding1DLayer;
                case 'ZeroPadding2D'
                    Obj = nnet.internal.cnn.keras.TranslatorForZeroPadding2DLayer;
                case 'TimeDistributedIn'
                    Obj = nnet.internal.cnn.keras.TranslatorForTimeDistributedLayerIn;
                case 'TimeDistributedOut'
                    Obj = nnet.internal.cnn.keras.TranslatorForTimeDistributedLayerOut;
                
                % Image Processing Toolbox: Resize layer
                case 'UpSampling2D' 
                    if nnet.internal.cnn.keras.isInstalledIPT
                        Obj = nnet.internal.cnn.keras.TranslatorForUpSampling2DLayer;
                    else
                        iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer', KerasLayerType, 'Image Processing Toolbox');
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                case 'UpSampling3D'
                    if nnet.internal.cnn.keras.isInstalledIPT
                        Obj = nnet.internal.cnn.keras.TranslatorForUpSampling3DLayer;
                    else
                        iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer', KerasLayerType, 'Image Processing Toolbox');
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                case 'Resizing'
                    if nnet.internal.cnn.keras.isInstalledIPT
                    Obj = nnet.internal.cnn.keras.TranslatorForResizingLayer;
                    else
                        iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer', KerasLayerType, 'Image Processing Toolbox');
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                              
                % Text Analytics Toolbox (TAT) layers:
                case 'Embedding'
                    if nnet.internal.cnn.keras.isInstalledTAT
                        iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:WordEmbeddingLayerIndexWarning');
                        Obj = nnet.internal.cnn.keras.TranslatorForEmbeddingLayer;
                    else
                        iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedProductLayer', KerasLayerType, 'Text Analytics Toolbox');
                        Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                    end
                % Unsupported layers:
                otherwise
                    Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
            end
            % Check whether translator can support layer-specific settings
            [tf, Message] = canSupportSettings(Obj, LSpec);
            if ~tf
                Obj = nnet.internal.cnn.keras.TranslatorForUnsupportedKerasLayer;
                iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedLayerSettingsWarning', KerasLayerType, getString(Message));
            end
            Obj.KerasLayerType = KerasLayerType;
        end
    end
    
    methods
        function S = importWeights(this, LayerName, SubmodelName, HDF5Filename, H5Info, TimeDistributedName)
            % Import weight parameters from the HDF5 file and return them
            % in the fields of struct S. If the architecture and weights
            % were saved, there will be a model_weights group. If only the
            % weights were saved, there will not be a model_weights group.
            
            % If there is a SubmodelName, then there are many layers'
            % weights in that group. But if there is no SubmodelName, then
            % the group contains only weights for the single layer. In
            % either case, the last bit of the weight path may not match
            % the layer name.
            if nargin < 6
                TimeDistributedName = '';
            end
            S = [];
            noSubmodel = isempty(SubmodelName);
            if hasModelWeightsGroup(H5Info)
                ModelWeightsPath = '/model_weights/';
            else
                ModelWeightsPath = '/';
            end
            if noSubmodel
                SubmodelName = LayerName;
            end
            SubmodelsWithWeights = trimNullChars(h5readatt(HDF5Filename, ModelWeightsPath, 'layer_names'));
            if ismember(SubmodelName, SubmodelsWithWeights)
                H5WeightInfo = h5info(HDF5Filename, [ModelWeightsPath SubmodelName]);
                SubmodelWeightNames = H5WeightInfo.Attributes.Value;
                if noSubmodel
                    WeightNamesThisLayer = SubmodelWeightNames;
                else
                    WeightNamesThisLayer = findWeightNamesThisLayer(SubmodelWeightNames, LayerName, SubmodelName, TimeDistributedName);
                end
                % Import weights
                if isequal(this.KerasLayerType, 'Bidirectional')
                    % Handle bidirectional LSTM as a special case
                    for i = 1:numel(WeightNamesThisLayer)
                        Path = WeightNamesThisLayer{i};
                        PathSplit = strsplit(Path, '/'); 
                        lstmName = PathSplit{2}; 
                        FullPath = sprintf('%s%s/%s', ModelWeightsPath, SubmodelName, Path);
                        try
                            ParameterValue = h5read(HDF5Filename, FullPath);
                        catch ME
                            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FailedToReadWeightParameter', FullPath)));
                        end
                        
                        ParameterName = PathSplit{end}; 
                        colpos = strfind(ParameterName, ':');
                        ParameterName = ParameterName(1:colpos-1);
                        if startsWith(lstmName, 'forward_lstm') 
                            ParameterName = ['forward_lstm_', ParameterName];
                        elseif startsWith(lstmName, 'backward_lstm')
                            ParameterName = ['backward_lstm_', ParameterName];
                        elseif startsWith(lstmName, 'forward_cu_dnnlstm') 
                            ParameterName = ['forward_cu_dnnlstm_', ParameterName];
                        elseif startsWith(lstmName, 'backward_cu_dnnlstm')
                            ParameterName = ['backward_cu_dnnlstm_', ParameterName];
                        else
                            error(message('nnet_cnn_kerasimporter:keras_importer:UnexpectedUnderlyingBidirectionalLayer'));
                        end
                        S.(ParameterName) = ParameterValue;
                    end
                else
                    % All layers except bidirectional LSTM
                    for i = 1:numel(WeightNamesThisLayer)
                        Path = WeightNamesThisLayer{i};
                        slashpos = strfind(Path, '/');
                        slashpos = slashpos(end);        
                        colonpos = strfind(Path, ':');
                        ParameterName = Path(slashpos+1 : colonpos-1);
                        FullPath = sprintf('%s%s/%s', ModelWeightsPath, SubmodelName, Path);
                        try
                            ParameterValue = h5read(HDF5Filename, FullPath);
                        catch ME
                            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FailedToReadWeightParameter', FullPath)));
                        end
                        S.(ParameterName) = ParameterValue;
                    end
                end
            end
        end
                
        function sz = fixKeras2DSizeParameter(~, szIn)
            % Convert scalar or column to row.
            if isscalar(szIn)
                sz = [szIn szIn];
            else
                assert(iscolumn(szIn));
                sz = szIn';
            end
        end
        
        function sz = fixKeras3DSizeParameter(~, szIn)
        % Convert scalar or column to row.
            if isscalar(szIn)
                sz = [szIn szIn szIn];
            else
                assert(iscolumn(szIn));
                sz = szIn';
            end
        end

        function strides = fixKerasStridesAllowNone(this, stridesIn, default)
            % Convert 'None' to default. Convert scalar or column to row.
            if isequal(stridesIn, 'None')
                strides = default;
            else
                strides = fixKeras2DSizeParameter(this, stridesIn);
            end
        end
        
        function strides = fixKeras3DStridesAllowNone(this, stridesIn, default)
            % Convert 'None' to default. Convert scalar or column to row.
            if isequal(stridesIn, 'None')
                strides = default;
            else
                strides = fixKeras3DSizeParameter(this, stridesIn);
            end
        end
        
    end
end

function WeightNamesThisLayer = findWeightNamesThisLayer(WeightNames, LayerName, SubmodelName, TimeDistributedName)
% Return (trimmed) names that match LayerName before the first slash if
% there's only one slash, else match LayerName before the last slash
WeightNamesThisLayer = {};
for i = 1:numel(WeightNames)
    FullName = WeightNames{i};
    slashpos = strfind(FullName, '/');
    if numel(slashpos) == 1
        slashpos = slashpos(1);
        Prefix = FullName(1:slashpos-1);
        if isequal(Prefix, LayerName) || isequal(Prefix, TimeDistributedName)
            WeightNamesThisLayer(end+1) = trimNullChars({FullName});
        end
    else
        slashpos1 = slashpos(end-1);
        slashpos2 = slashpos(end);
        Prefix = FullName(slashpos1+1: slashpos2-1);
        if isequal(Prefix, LayerName)
            WeightNamesThisLayer(end+1) = trimNullChars({FullName});
        end
    end
%     if isequal(Prefix, LayerName) ||...
%             (startsWith(SubmodelName, 'time_distributed') &&...
%             isequal(Prefix, SubmodelName) && ~startsWith(LayerName, 'time_distributed'))
%         WeightNamesThisLayer(end+1) = trimNullChars({FullName});
%     end
end
end

function cellStr = trimNullChars(cellStr)
% Remove null chars (for which double(ch)==0) from all char arrays in
% cellStr.
for i = 1:numel(cellStr)
    cellStr{i}(cellStr{i}==0) = [];
end
end

function tf = hasModelWeightsGroup(H5Info)
tf = any(arrayfun(@(Grp)isequal(Grp.Name, '/model_weights'), H5Info.Groups));
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end

function name = legalizeParameterName(name)
% Replace slashes with a special delimiter
delim = '___REPLACEDSLASH___';
name = strrep(name, '/', delim);
end

function tf = isBidirectionalLayer(LSpec)
tf = isequal(LSpec.Type, 'Bidirectional') && ...
    isequal(LSpec.KerasConfig.merge_mode, 'concat');    
end