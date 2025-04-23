classdef TranslatorForDAGInputLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2017-2021 The MathWorks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            %   struct with fields:
            %                  name: 'input_1'
            %                 dtype: 'float32'
            %     batch_input_shape: [4x1 double] or [5x1 double]
            %                sparse: 0
            if hasKerasField(LSpec, 'batch_input_shape')
                KerasInputShape = kerasField(LSpec, 'batch_input_shape');
            elseif ~isempty(UserImageInputSize)
                KerasInputShape = [NaN UserImageInputSize(:)'];
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ImageInputSizeNeeded')));
            end
            
            % Check unsupported options
            if kerasField(LSpec, 'sparse') ~= 0
                iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:NoSparseInput', LSpec.Name);
            end
            if  ~LSpec.IsSequenceInput
                switch numel(KerasInputShape)
                    case 2
                        % Each input pattern is just a vector. Use the [height,width] syntax
                        % for the first argument to imageInputLayer. Make the vector a "row image":
                        NNTInputShape = KerasInputShape(2);
                    case 3
                        % Assume it's an 2D image input without channels
                        % specified: [NaN height width], implying 1 channel.
                        NNTInputShape = nnet.internal.cnn.keras.determineImageInputSize(KerasInputShape, UserImageInputSize);
                    case 4
                        % Assume it's an 2D image input layer:
                        % Keras inputShape: (NaN, rows, cols, channels)
                        NNTInputShape = nnet.internal.cnn.keras.determineImageInputSize(KerasInputShape, UserImageInputSize);
                    case 5
                        % Assume it's a 3D image input layer:
                        % Keras inputShape: (NaN, rows, cols, depth, channels)
                        NNTInputShape = nnet.internal.cnn.keras.determineImageInputSize(KerasInputShape, UserImageInputSize);
                    otherwise
                        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:BadInputDim', num2str(numel(KerasInputShape)-1))));
                end
                if any(isnan(NNTInputShape))
                    % The input layer has insufficient input size
                    % information. a placeholder layer for the input 
                    % layer is used.  
                    NNTLayers  = { nnet.keras.layer.PlaceholderInputLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name), NNTInputShape) };
                else
                    if numel(NNTInputShape) > 3 
                        % NNTInputShape is of the form [h w d c]
                        NNTLayers = { image3dInputLayer(NNTInputShape(:)', 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name), 'Normalization', 'none') };                    
                    elseif numel(NNTInputShape) > 1
                        % NNTInputShape is of the form [h w c]
                        NNTLayers = { imageInputLayer(NNTInputShape(:)', 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name), 'Normalization', 'none') };
                    else
                        %%NNTInputShape is of the form [c]
                        NNTLayers = { featureInputLayer(NNTInputShape, 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
                    end
                end
            else
                if numel(KerasInputShape) == 5
                    % This is the case where the input layer is outside of
                    % the timeDistributed wrapper but we still want to
                    % create a sequence input layer.
                    ShouldBeNaN	= KerasInputShape(1);
                    Timesteps   = KerasInputShape(2);
                    Height      = KerasInputShape(3);
                    Width       = KerasInputShape(4);
                    Channel     = KerasInputShape(5);
                elseif numel(KerasInputShape) == 4
                    % This is the case where the input layer is inside of
                    % the timeDistributed wrapper and we would like to
                    % create a sequence input layer.   
                    ShouldBeNaN	= KerasInputShape(1);
                    Timesteps   = nan;
                    Height      = KerasInputShape(2);
                    Width       = KerasInputShape(3);
                    Channel     = KerasInputShape(4);
                elseif numel(KerasInputShape) == 3
                    % This is the case when we have an input layer with 
                    % batch size, timesteps and input_size and we would
                    % like to create a sequence input layer
                    ShouldBeNaN	= KerasInputShape(1);
                    Timesteps   = KerasInputShape(2);
                    Height      = [];
                    Width       = [];
                    Channel     = KerasInputShape(3);
                elseif numel(KerasInputShape) == 2
                    Height      = [];
                    Width       = [];
                    Channel     = 1; 
                else
                    assert(false);
                end
                if (LSpec.Has1DLayers && ~isnan(Timesteps))
                    NNTLayers       = {sequenceInputLayer([Height, Width, Channel], 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name),'MinLength',Timesteps)};
                else
                    NNTLayers       = {sequenceInputLayer([Height, Width, Channel], 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name))};
                end
            end
        end
        
        function [tf, Message] = canSupportSettings(~, LSpec)
            tf = true;
            Message = '';
            if hasKerasField(LSpec, 'batch_input_shape')
                InputShape = kerasField(LSpec, 'batch_input_shape');
                if ~ismember(numel(InputShape), [2 3 4 5])
                    tf = false;
                    Message = message('nnet_cnn_kerasimporter:keras_importer:BadInputDim', num2str(numel(InputShape)-1));
                end
            end
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end
