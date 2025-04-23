function FolderPath = validateTFModelFolder(FolderPath)
    % Copyright 2021-2022 The MathWorks, Inc.  
    FolderPath = char(FolderPath);
    folderExists = isfolder(FolderPath);
    fileExists = isfile(FolderPath);
    if ~folderExists && ~fileExists
        % neither a folder nor a file exists at the given path
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FolderNotFound', FolderPath)));
    elseif ~folderExists && fileExists
        % a folder does not exist but a file exists at the given path
        % check if trying to import an older keras HDF5 model file
        maybeKerasFormat = nnet.internal.cnn.tensorflow.util.checkKerasFileFormat(FolderPath);
        if maybeKerasFormat
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FolderNotFoundButH5FileExistsImportLayers', FolderPath)));
        else
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FolderNotFound', FolderPath)));
        end
    elseif folderExists
        % a folder exists, we should check if there is a 'saved_model.pb' file and a 'variables' sub folder
        if ~isfile([FolderPath filesep 'saved_model.pb'])
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:SavedModelPbNotFound', FolderPath)));
        end
        if ~isfolder([FolderPath filesep 'variables'])
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:VariablesNotFound', FolderPath)));
        end
    end
end