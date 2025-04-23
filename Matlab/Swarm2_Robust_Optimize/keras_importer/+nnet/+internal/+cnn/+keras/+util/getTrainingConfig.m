function trainingConfig = getTrainingConfig(outputLayerType)
% Copyright 2021 The MathWorks, Inc.
% Create a dummy TrainingConfig that specifies an output layer of the right type
        trainingConfig = struct;
		switch outputLayerType
			case {'classification', 'pixelclassification'}
				trainingConfig.loss = 'categorical_crossentropy';
			case 'regression'
				trainingConfig.loss = 'mse';
			case 'binarycrossentropyregression'
				trainingConfig.loss = 'binary_crossentropy';
			otherwise
				throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:noCVSTForPixelClassification')));
		end

end

