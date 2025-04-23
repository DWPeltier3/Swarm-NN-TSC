classdef ConverterForUnsupportedLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % A Keras custom layer definition will be generated. The number of
    % input and output variables of the Keras layer's 'call' method will
    % match the number of incoming connections of the MATLAB layer.
    % Learnable parameters in the MATLAB layer will be saved as weights, to
    % be loaded by the Keras layer. Because the shape of the weights can
    % vary across different instantiations of the layer in the network, the
    % MATLAB size of the Learnable parameters will be made hyperparameters
    % of the Keras layer, and they will appear as arguments to the Keras
    % layer constructor. It's the MATLAB size, because we don't know the
    % meaning of the Learnable parameters' dimensions and therefore cannot
    % permute them to a standard Keras ordering. The code in the Keras
    % model file that creates each instance of the layer will pass the
    % correct size for each instance.
    %
    % For example: A MATLAB preluLayer with Learnable parameter Alpha might
    % export to this Keras custom layer definition:
    %
    % class TFExporter_preluLayer(tf.keras.layers.Layer):
    %     def __init__(self, Alpha_Shape_=(), name=None):
    %         super(TFExporter_preluLayer, self).__init__(name=name)
    %         # Learnable parameters:
    %         self.Alpha = tf.Variable(name="Alpha", initial_value=tf.zeros(Alpha_Shape_), trainable=True)
    %         # Add code to implement the layer constructor here.
    %
    %     def call(self, input1):
    %         # Add code to implement the layer's forward pass here.
    %         # The input tensor format(s) are: BSSC
    %         # The output tensor format(s) are: BSSC
    %         # where B=batch, C=channels, T=time, S=spatial(in order of height, width, depth,...)
    %         output1 = tf.math.maximum(input1, 0.0) + self.Alpha * tf.math.minimum(0.0, input1);
    %         return output1
    %
    %
    % The Keras model file would contain calls that pass the shapes like
    % this:
    %
    % def create_model():
    %     image3chan = keras.Input(shape=(2,5,3))
    %     image7chan = keras.Input(shape=(2,5,7))
    %     prelu3chan = TFExporter_preluLayer(name="prelu3chan", Alpha_Shape_=(1,1,3))(image3chan)
    %     prelu7chan = TFExporter_preluLayer(name="prelu7chan", Alpha_Shape_=(1,1,7))(image7chan)
    %     concat = layers.Concatenate(axis=3)([prelu3chan, prelu7chan])
    %
    %     model = keras.Model(inputs=[image3chan, image7chan], outputs=[concat])
    %     return model

    methods
        function convertedLayer = toTensorflow(this)
            layerPieces = split(class(this.Layer), '.');
            layerType = layerPieces{end};
            % gen code
            actualInputParamStr = join(this.InputTensorName, ", ");
            actualOutputParamStr = join(this.OutputTensorName, ", ");
            formalInputParamStr = join("input" + string(1:numel(this.InputTensorName)), ", ");
            formalOutputParamStr = join("output" + string(1:numel(this.OutputTensorName)), ", ");

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            layerClassName = layerType;

            % If the layer has learnables and/or state, name the layer and
            % export the Learnables and/or state as weights:

            verifyNumericLearnablesAndState(this);
            msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForUnsupportedLayer", this.Layer.Name, class(this.Layer));
            warningNoBacktrace(this, msg);
            convertedLayer.WarningMessages(end+1) = msg;

            LearnablesTbl = LearnableParametersTable(this.layerAnalyzer);
            StateTbl = StateParametersTable(this.layerAnalyzer);
            nameClause = "";
            weightDeclarationLines = string.empty;
            if ~isempty(LearnablesTbl) || ~isempty(StateTbl)
                layerName = this.OutputTensorName(1) + "_";
                nameClause = "name=""" + layerName + """";
                convertedLayer.LayerName = layerName;
            end
            % Generate code to allocate space for the learnables (weights)
            % in the Keras layer definition
            weightShapeDefClause = "";
            weightShapeCallClause = "";
            if ~isempty(LearnablesTbl)
                weightDeclarationLines(end+1,1) = "        # Learnable parameters: These have been exported from MATLAB and will be loaded automatically from the weight file:";
                VarNames = string(LearnablesTbl.Parameter);
                for i=1:numel(VarNames)
                    varName = VarNames(i);
                    shape = LearnablesTbl{varName, "Size"}{1};
                    colMajWeight = this.Layer.(varName);
                    rowMajWeight = permute(colMajWeight, numel(shape):-1:1);
                    convertedLayer.weightNames(end+1) = varName;
                    convertedLayer.weightArrays{end+1} = rowMajWeight;
                    convertedLayer.weightShapes{end+1} = shape;
                    shapeStr = "(" + join(string(shape), ',') + ")";
                    weightShapeName = varName + "_Shape_";
                    weightShapeDefClause = weightShapeDefClause + ", " + weightShapeName + "=None";
                    weightShapeCallClause = weightShapeCallClause + ", " + weightShapeName + "=" + shapeStr;
                    weightDeclarationLines(end+1,1) = "        self." + varName + " = tf.Variable(name=""" + varName + """, initial_value=tf.zeros(" + weightShapeName + "), trainable=True)"; %#ok<*AGROW>
                end
            end
            % Generate code to allocate space for the state variables
            % (treated as weights) in the Keras layer definition
            if ~isempty(StateTbl)
                weightDeclarationLines(end+1,1) = "        # State parameters:";
                VarNames = string(StateTbl.Parameter);
                for i=1:numel(VarNames)
                    varName = VarNames(i);
                    shape = StateTbl{varName, "Size"}{1};
                    colMajWeight = this.Layer.(varName);
                    rowMajWeight = permute(colMajWeight, numel(shape):-1:1);
                    convertedLayer.weightNames(end+1) = varName;
                    convertedLayer.weightArrays{end+1} = rowMajWeight;
                    convertedLayer.weightShapes{end+1} = shape;
                    shapeStr = "(" + join(string(shape), ',') + ")";
                    weightShapeName = varName + "_Shape_";
                    weightShapeDefClause = weightShapeDefClause + ", " + weightShapeName + "=()";
                    weightShapeCallClause = weightShapeCallClause + ", " + weightShapeName + "=" + shapeStr;
                    weightDeclarationLines(end+1,1) = "        self." + varName + " = tf.Variable(name=""" + varName + """, initial_value=tf.zeros(" + weightShapeName + "), trainable=False)";
                end
            end
            % Assemble the final Keras layer definition code
            inputTensorFormats = join(nnet.internal.cnn.tf_exporter.FormatConverter.mlFormatToTfFormat(this.InputFormat), ", ");
            outputTensorFormats = join(nnet.internal.cnn.tf_exporter.FormatConverter.mlFormatToTfFormat(this.OutputFormat), ", ");
            convertedLayer.layerCode = sprintf("%s = %s(%s%s)(%s)", actualOutputParamStr, layerClassName, nameClause, weightShapeCallClause, actualInputParamStr);
            convertedLayer.placeholderLayerName = layerClassName;
            convertedLayer.placeholderLayerCode = [
                "import tensorflow as tf"
                "import sys     # Remove this line after completing the layer definition."
                ""
                "class " + layerClassName + "(tf.keras.layers.Layer):"
                "    # Add any additional layer hyperparameters to the constructor's"
                "    # argument list below."
                "    def __init__(self" + weightShapeDefClause + ", name=None):"
                "        super(" + layerClassName + ", self).__init__(name=name)"
                weightDeclarationLines
                ""
                "    def call(self, " + formalInputParamStr + "):"
                "        # Add code to implement the layer's forward pass here."
                "        # The input tensor format(s) are: " + inputTensorFormats
                "        # The output tensor format(s) are: " + outputTensorFormats
                "        # where B=batch, C=channels, T=time, S=spatial(in order of height, width, depth,...)"
                ""
                "        # Remove the following 3 lines after completing the custom layer definition:"
                "        print(""Warning: load_model(): Before you can load the model, you must complete the definition of custom layer " + layerClassName + " in the customLayers folder."")"
                "        print(""Exiting..."")"
                "        sys.exit(""See the warning message above."")"
                ""
                "        return " + formalOutputParamStr
                ""
                ];
        end

        function verifyNumericLearnablesAndState(this)
            % Warn if the layer has any Learnable or State
            % variables that are not numeric.
            meta = metaclass(this.Layer);
            PropertyList = meta.PropertyList;
            for i=1:numel(PropertyList)
                isLearnableOrState = false;
                try
                    isLearnableOrState = PropertyList(i).Learnable || PropertyList(i).State;
                catch
                    % Ignore errors. .Learnable and .State may not exist.
                end
                if isLearnableOrState
                    name = PropertyList(i).Name;
                    if ~isnumeric(this.Layer.(name))
                        msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForUnsupportedLayer2", this.Layer.Name);
                        error(msg);
                    end
                end
            end
        end
    end
end