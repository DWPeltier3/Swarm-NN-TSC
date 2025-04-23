classdef ConverterForFullyConnectedLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = string.empty;

            % The keras Dense layer operates only on the last input
            % dimension. In contrast, the DLT FC layer first flattens its
            % S*C input dimensions into a single C and then performs the
            % FC. In DAGNetworks, it then re-expands it to 1*C.

            % Support table of cases that occur across the FC layer in DLT:
            %
            % Non-temporal cases:
            %     DLT formats          TF formats       TF layers               Occurs in
            %  ---------------      ---------------     ---------               ---------
            % 1   S..CB --> 1..CB    BS..C --> B1..C     Reshape(1..C),Dense     DAGnet
            % 2      CB --> CB       BC --> BC           Dense                   DAGnet, dlnet
            % 3   S..CB --> CB       BS..C --> BC        Reshape(C),Dense        dlnet
            % 4    S..C --> CB       BS..C --> BC        Reshape(C),Dense        dlnet
            % 4.1    CU --> CB       BC --> BC           Dense                   dlnet      Only suppoting U=1

            % Temporal cases: (Just wrap individial operations in TimeDistributed as needed)
            %  DLT formats             TF formats          TF layers                       Occurs in
            %  ---------------         ---------------     ---------                       ---------
            % 5   S..CBT --> 1..CBT    BTS..C --> BT1..C   [TimeDistributed(Reshape(1..C)) DAGnet
            %                                               TimeDistributed(Dense)]
            % --> Case 5 does not exist! You can't apply FC to an S*T tensor in
            % a DAGNetwork. The error says: "Layer 'fc': Invalid input data
            % for fully connected layer. The input data must not have both
            % spatial and temporal dimensions."
            %
            % 6      CBT --> CBT       BTC --> BTC         Dense                           DAGnet, dlnet
            % 7   S..CBT --> CBT       BTS..C --> BTC      [TimeDistributed(Reshape(C))    dlnet
            %                                               TimeDistributed(Dense)]
            % 8       CT --> CT        BTC --> BTC         Dense                           dlnet
            % 9    S..CT --> CT        BTS..C --> BTC      [TimeDistributed(Reshape(C))    dlnet
            %                                               TimeDistributed(Dense)]

            inSize = this.InputSize{1};                      % excludes BT dimensions
            outSize = this.OutputSize{1};                    % excludes BT dimensions
            numOutputFeatures = outSize(end);

            % (1) Maybe add a pre-flattening layer (possibly TimeDistributed):
            didFlatten = false;
            needTimeDistributed = false;
            if numel(inSize) > 1
                % Input is S..C, so must pre-flatten.
                needTimeDistributed = this.layerAnalyzer.IsTemporal;
                flattenName1 = this.OutputTensorName + "_preFlatten1";
                numSpatial = numel(inSize)-1;
                if numel(outSize) > 1
                    % S..C --> 1..C
                    shape = "(" + join(string(ones(1,numSpatial)), ", ") + ", -1)";
                else
                    % S..C --> C
                    shape = "(-1,)";
                end
                line = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Reshape", "%s", {shape}, flattenName1, needTimeDistributed);
                convertedLayer.layerCode = [convertedLayer.layerCode; line];
                didFlatten = true;
            end

            % (2) Add the Dense layer (possibly TimeDistributed):
            layerName = this.OutputTensorName + "_";
            if didFlatten
                from = this.OutputTensorName;
            else
                from = this.InputTensorName;
            end
            line = kerasCodeLine(this, from, this.OutputTensorName, "layers.Dense", "%d", {numOutputFeatures}, layerName, needTimeDistributed);
            convertedLayer.layerCode = [convertedLayer.layerCode; line];

            % (3) Create tf weights:
            if ~isempty(this.Layer.Weights)
                % We need to rearrange the weight indices so that they do
                % the right thing in keras after the row-major flattening
                % that occurs there
                if numel(inSize)==1
                    inSize = [inSize 1];
                end
                weightIndices = permute(reshape(1:prod(inSize), inSize), numel(inSize):-1:1);
                weightIndices = weightIndices(:);
                kerasWeights = this.Layer.Weights(:, weightIndices);               % Don't transpose, because AxB in colmajor is already BxA in rowmajor
                convertedLayer.weightNames     = ["kernel", "bias"];
                convertedLayer.weightArrays = {kerasWeights; this.Layer.Bias};                        % A matrix and a vector.
                convertedLayer.weightShapes = {size(kerasWeights'); numel(this.Layer.Bias)};           % 2D and 1D.
                convertedLayer.LayerName = layerName;
            end
        end
    end
end
