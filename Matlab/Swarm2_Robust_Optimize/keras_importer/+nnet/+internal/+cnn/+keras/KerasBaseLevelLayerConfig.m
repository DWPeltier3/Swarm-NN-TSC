classdef KerasBaseLevelLayerConfig < handle

% Copyright 2017 The Mathworks, Inc.

    properties
        KerasStruct     % The raw imported Keras struct defining the layer
    end
    
    methods
        function this = KerasBaseLevelLayerConfig(Struct)
            % Struct =
            %   struct with fields:
            %
            %                     name: 'conv2d_1'
            %        kernel_constraint: []
            %        batch_input_shape: [4×1 double]
            %            dilation_rate: [2×1 double]
            %         bias_initializer: [1×1 struct]
            %                    dtype: 'float32'
            %                 use_bias: 1
            %                trainable: 1
            %              kernel_size: [2×1 double]
            %         bias_regularizer: []
            %     activity_regularizer: []
            %               activation: 'linear'
            %          bias_constraint: []
            %              data_format: 'channels_last'
            %                  strides: [2×1 double]
            %                  filters: 8
            %       kernel_regularizer: []
            %       kernel_initializer: [1×1 struct]
            %                  padding: 'valid'
            this.KerasStruct = Struct;
        end
    end
end

