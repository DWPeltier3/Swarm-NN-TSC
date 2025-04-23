classdef ConverterForWordEmbeddingLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022-2023 The MathWorks, Inc.

    % Supported DLT input formats: CBT, CT, CB.

    % Important features:
    %
    % (1) Vocabulary size: The MATLAB wordEmbeddingLayer automatically adds
    % an extra entry to the vocabulary that means "other". Indices into the
    % layer that are 0 or exceed the vocabulary size are mapped to this
    % input. It's stored as an extra column on the end of the weight
    % matrix. Keras doesn't have that, so we implement it by inserting a TF
    % custom layer into the model that does the mapping of 0 or large
    % indices to the last word in the extended vocab. See
    % PrepareEmbeddingLayer.py.
    %
    % (2) Origin-1 indexing: MATLAB uses origin-1 indexing into the
    % wordEmbeddingLayer. The exported TF model needs to work on the same
    % input data but performs origin-0 indexing. So the custom layer also
    % subtracts 1 from the input index before accessing the TF weight
    % matrix.
    %
    % (3) Pre- and post-processing in keras: The keras Embedding layer
    % performs the mapping BT-->BTC. The MATLAB wordEmbeddingLayer performs
    % CBT-->CBT, CT-->CT, or CB-->CB. For the CBT and CT cases, we add a
    % preprocessing keras layer to produce BT from BTC. For the CB case, BC
    % is used in TF and is interpreted by the Embedding layer as BT, and we
    % add postprocessing to transform the layer's output from BTC back to
    % BT to represent MATLAB's CB.


    methods
        function convertedLayer = toTensorflow(this)
            if this.Layer.OOVMode=="map-to-last"
                accept_oov = "True";
                input_dim = this.Layer.NumWords + 1;
            else
                accept_oov = "False";
                input_dim = this.Layer.NumWords;
            end

            output_dim = this.Layer.Dimension;
            
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            switch this.InputFormat
                case {"CBT", "CT"}
                    doSqueeze = "True";
                    postprocessingLine = string.empty;
                case {"CB"}
                    doSqueeze = "False";
                    postprocessingLine = sprintf("%s = tf.squeeze(%s, 1)", this.OutputTensorName, this.OutputTensorName); % BTC=B1C --> BC
                otherwise
                    convertedLayer.Success = false;
                    return
            end

            layerName = this.OutputTensorName + "_";
            convertedLayer.layerCode = [
                % preprocessing
                kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, ...
                "PrepareEmbeddingLayer", "%d., %s, %s", {input_dim, doSqueeze, accept_oov}, ...
                "", false)

                % embedding layer
                kerasCodeLine(this, this.OutputTensorName, this.OutputTensorName, ...
                "layers.Embedding", "%d, %d", {input_dim, output_dim}, ...
                layerName, false)

                % postprocessing
                postprocessingLine
                ];

            % TF weight shape is [input_dim, output_dim]. MATLAB weight
            % size is [output_dim, input_dim].
            kerasWeights = this.Layer.Weights;
            % kerasWeights are already in the correct TF linear memory
            % ordering because the dimension ordering is flipped between TF
            % and MATLAB.
            convertedLayer.LayerName    = layerName;
            convertedLayer.weightNames  = "embeddings";
            convertedLayer.weightArrays = {kerasWeights};
            convertedLayer.weightShapes = {[input_dim, output_dim]};
            convertedLayer.customLayerNames = "PrepareEmbeddingLayer";
        end
    end
end
