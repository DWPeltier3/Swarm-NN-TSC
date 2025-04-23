function checkOutputLayerTypeAndClasses(isRegression, PassedClasses, isPixelClassification, isRNN, outputTensorSize)
    % Copyright 2021-2023 The MathWorks, Inc.
    if isRegression
        % It's a regression layer
        if ~nnet.internal.cnn.keras.util.isAuto(PassedClasses)
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ClassesForRegression')));
        end
    elseif isPixelClassification
        % It's a pixelClassification layer
        if ~nnet.internal.cnn.keras.isInstalledCVST
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:noCVSTForPixelClassification')));
        end
    elseif nnet.internal.cnn.keras.util.isAuto(PassedClasses)
        % 'auto' classes passed. assembleNetwork will set them later.
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:FillingInClasses');									
    else
        % Classes passed.
        % Canonicalize the passed classes
        PassedClasses = nnet.internal.cnn.layer.paramvalidation.convertClassesToCanonicalForm(PassedClasses);

        if isRNN
            netNumClasses = outputTensorSize(1);
        else
            netNumClasses = outputTensorSize(end);
        end
        
        % Verify if the user passed the right number of classes
        if netNumClasses ~= numel(PassedClasses)
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ClassesMismatchSize', netNumClasses, numel(PassedClasses))));
        end
    end
end
