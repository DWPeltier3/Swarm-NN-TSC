function WeightFile = validateKerasWeightFile(WeightFile)
    % Copyright 2021 The MathWorks, Inc.
    if ~(isa(WeightFile,'char') || isa(WeightFile,'string'))
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:WeightFileString')));
    end
    if ~isempty(WeightFile)
        WeightFile = char(WeightFile);
        if ~exist(WeightFile, 'file')
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FileNotFound', WeightFile)));
        end
    end
end