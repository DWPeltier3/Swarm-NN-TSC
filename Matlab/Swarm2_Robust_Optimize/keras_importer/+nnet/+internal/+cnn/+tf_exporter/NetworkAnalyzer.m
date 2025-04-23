classdef NetworkAnalyzer < handle

    %   Copyright 2022 The MathWorks, Inc.

    properties (SetAccess=protected)
        Net
        LayerAnalyzers  nnet.internal.cnn.tf_exporter.LayerAnalyzer
    end

    properties (Dependent)
        InputNames  string
        OutputNames string
    end

    properties (Access=protected)
        InternalNetworkAnalyzer nnet.internal.cnn.analyzer.NetworkAnalyzer
    end

    methods
        function this = NetworkAnalyzer(net)
            this.Net = net;
            this.InternalNetworkAnalyzer = nnet.internal.cnn.analyzer.NetworkAnalyzer(net);
            this.LayerAnalyzers = arrayfun(@nnet.internal.cnn.tf_exporter.LayerAnalyzer, this.InternalNetworkAnalyzer.LayerAnalyzers);
        end

        function names = get.InputNames(this)
            names = string(this.Net.InputNames);
        end

        function names = get.OutputNames(this)
            names = string(this.Net.OutputNames);
        end
    end
end
