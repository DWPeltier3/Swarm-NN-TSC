classdef TranslatorForMergeLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2017 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %         mode_type: 'raw'
            %              name: 'mbox_conf'
            %              mode: 'concat'
            % output_shape_type: 'raw'
            %      output_shape: []
            %       concat_axis: 1
            %  output_mask_type: 'raw'
            %       output_mask: []
            %         arguments: [1×1 struct]
            %          dot_axes: -1
            NumInputs = numel(LSpec.InConns);
            switch kerasField(LSpec, 'mode')
                case 'sum'
                    NNTLayers = { additionLayer(NumInputs, 'Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
                case 'concat'
                    if LSpec.IsFeatureInput
                         NNTLayers = { concatenationLayer(1, NumInputs, 'Name', LayerName) }; 
                    else
                        NNTLayers = { depthConcatenationLayer(NumInputs, 'Name', LSpec.Name) };
                    end
                otherwise
                    assert(false)
            end
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if ~ismember(kerasField(LSpec, 'mode'), {'sum', 'concat'})
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedMergeMode', LSpec.Name, kerasField(LSpec, 'mode'));
            elseif ~ismember(kerasField(LSpec, 'concat_axis'), [-1 1 3])
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedAxis', LSpec.Name);
            elseif ~isempty(kerasField(LSpec, 'output_shape'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedMergeOutputShape', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end

