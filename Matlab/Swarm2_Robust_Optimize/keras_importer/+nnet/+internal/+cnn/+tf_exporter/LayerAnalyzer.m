classdef LayerAnalyzer < handle

    %   Copyright 2022 The MathWorks, Inc.

    properties (SetAccess=protected)
        Name
        ExternalLayer
        NumInputs       double
        NumOutputs      double
        OutputNames     string
        IsCustomLayer   logical
        IsTemporal      logical     % true if the first input tensor format contains a T
        IsInputLayer    logical
        IsOutputLayer   logical
    end

    properties (Access=protected)
        InternalLayerAnalyzer nnet.internal.cnn.analyzer.util.LayerAnalyzer
    end

    methods
        function this = LayerAnalyzer(internalLayerAnalyzer)
            this.InternalLayerAnalyzer = internalLayerAnalyzer;
            this.Name = this.InternalLayerAnalyzer.Name;
            this.ExternalLayer =  this.InternalLayerAnalyzer.ExternalLayer;
            if isa(this.ExternalLayer, 'nnet.layer.ClassificationLayer') 
                % These are the custom output layers
                this.NumInputs = 1;
                this.NumOutputs = 1;
                this.OutputNames = string(this.ExternalLayer.Name);
            elseif isa(this.ExternalLayer, 'nnet.layer.RegressionLayer')
                % These are the custom output layers
                this.NumInputs = 1;
                this.NumOutputs = 1;
                this.OutputNames = string(this.ExternalLayer.Name) + "/" + string(this.ExternalLayer.ResponseNames);
            else
                this.NumInputs = this.ExternalLayer.NumInputs;
                this.NumOutputs = this.ExternalLayer.NumOutputs;
                this.OutputNames = string(this.ExternalLayer.Name) + "/" + string(this.ExternalLayer.OutputNames);
            end
            this.IsTemporal     = this.NumInputs > 0 && contains(inputFormat(this, 1), "T");
            this.IsCustomLayer = this.InternalLayerAnalyzer.IsCustomLayer;
            this.IsInputLayer =  this.InternalLayerAnalyzer.IsInputLayer;
            this.IsOutputLayer = this.InternalLayerAnalyzer.IsOutputLayer;
        end

        %% Layer input properties
        function name = inputSourceName(this, inputNum)
            % Returns the input source for inputNum. The source is a DLT
            % full port name such as "layerName" or "layerName/portName".
            % If the input is a dlnetwork input with no input layer, then
            % the source string will be "/n" where n is the number of the
            % network input (including both those with ans without input
            % layers).
            name = string(this.InternalLayerAnalyzer.Inputs.Source{inputNum});
        end

        function sz = inputSize(this, inputNum)
            sz = this.InternalLayerAnalyzer.Inputs.Size{inputNum};
        end

        function fmt = inputFormat(this, inputNum)
            fmt = string(this.InternalLayerAnalyzer.Inputs.Meta{inputNum}.dims);
        end

        %% Layer output properties
        function sz = outputSize(this, outputNum)
            sz = this.InternalLayerAnalyzer.Outputs.Size{outputNum};
        end

        function fmt = outputFormat(this, outputNum)
            fmt = string(this.InternalLayerAnalyzer.Outputs.Meta{outputNum}.dims);
        end

        %         function ch = outputDestination(this, outputNum)
        %             % Returns the output destination for outputNum. The destination
        %             % is a DLT full port name such as "layerName" or
        %             % "layerName/portName".
        %             ch = this.InternalLayerAnalyzer.Outputs.Destination{outputNum};
        %         end
        %% Information about parameters
        function Tbl = LearnableParametersTable(this)
            Tbl = this.InternalLayerAnalyzer.Learnables;
        end

        function Tbl = StateParametersTable(this)
            Tbl = this.InternalLayerAnalyzer.Dynamics;
        end
    end
end
