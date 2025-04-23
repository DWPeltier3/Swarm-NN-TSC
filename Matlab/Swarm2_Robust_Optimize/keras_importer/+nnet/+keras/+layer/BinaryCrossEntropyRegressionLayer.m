classdef BinaryCrossEntropyRegressionLayer < nnet.layer.RegressionLayer
    properties
        BatchIdx
    end
    
    methods
        function this = BinaryCrossEntropyRegressionLayer(Name, isRNN)
            this.Name        = Name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:BinaryCrossEntropyRegressionLayerDescription'));
            this.Type        = getString(message('nnet_cnn_kerasimporter:keras_importer:BinaryCrossEntropyRegressionLayerType'));
            if isRNN
                % Y has size [c n t]
                this.BatchIdx = 2;
            else
                % Y has size [h w c n]
                this.BatchIdx = 4;
            end
        end
        
        function loss = forwardLoss(this, Y, T)
            Ybnd         = nnet.internal.cnn.util.boundAwayFromZero(Y);
            OneMinusYbnd = nnet.internal.cnn.util.boundAwayFromZero(1-Y);
            batchSize    = size(Y, this.BatchIdx);
            loss         = -sum(T.*log(Ybnd) + (1-T).*log(OneMinusYbnd), 'all') / batchSize;
        end
        
        function dLdY = backwardLoss(this, Y, T)
            Denom     	= nnet.internal.cnn.util.boundAwayFromZero(Y.*(1-Y));
            batchSize 	= size(Y, this.BatchIdx);
            dLdY        = ((Y-T)./Denom)/batchSize;
        end
    end
end