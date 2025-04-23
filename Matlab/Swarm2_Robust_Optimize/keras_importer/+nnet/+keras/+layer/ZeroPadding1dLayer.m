classdef ZeroPadding1dLayer <  nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.internal.cnn.layer.CPUFusableLayer
    % ZeroPadding1dLayer   ZeroPadding1dLayer layer
    %
    %   layer = ZeroPadding1dLayer(Name, Amounts) creates a layer with
    %   name Name that pads the input sequence with zeros.
    %
    %       layer = ZeroPadding1dLayer(Name, Pad), where Pad is a scalar
    %           integer, pads the left and right with the same
    %           number of columns.
    %
    %       layer = ZeroPadding1dLayer(Name, [Left, Right]),
    %           vector, pads  the left and right with 'Left' and 'Right' columns

    %   Copyright 2021-2023 The MathWorks, Inc.

    properties
        leftPad
        rightPad
    end
    
    methods
        function this = ZeroPadding1dLayer(name, Amounts)
            assert(all(Amounts >= 0));
            if isscalar(Amounts)
                this.leftPad    = Amounts;
                this.rightPad   = Amounts;
            elseif isvector(Amounts)
                if numel(Amounts)==2
                    this.leftPad  = Amounts(1);
                    this.rightPad = Amounts(2);
                end
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ZP2DAmounts')));
            end
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:ZeroPadding2dDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:ZeroPadding2dType'));
        end
        
        function Z = predict( this, X )
            if isequal(dims(X), 'CBT')
                % X is size [C B T]. Z is size [C B T+Left+Right]
                [C, B, T] = size(X);
                Z = zeros(C, B, T + this.leftPad + this.rightPad, 'like', X);
                Z(:,:,this.leftPad + (1:T)) = X;
            elseif isequal(dims(X), 'CT')
                % X is size [C T]. Z is size [C T+Left+Right]
                [C, T] = size(X);
                Z = zeros(C, T + this.leftPad + this.rightPad, 'like', X);
                Z(:, this.leftPad + (1:T)) = X;
            else
                % Error if X has unexpected data format
                throw(MException('nnet_cnn_kerasimporter:keras_importer:KerasInternalLayerUnsupportedFormat', ...
                    message('nnet_cnn_kerasimporter:keras_importer:KerasInternalLayerUnsupportedFormat', this.Name, 'CBT')));
            end
        end
        
    end

    methods(Hidden)
        function layerArgs = getFusedArguments(layer)
            %getFusedArguments  Return arguments needed to call the
            % layer in a fused network
            layerArgs = { 'zeropad', [layer.leftPad layer.rightPad] };
        end

        function tf = isFusable(~)
            %isFusable Flag if layer is fusable
            tf = true;
        end
    end
end
