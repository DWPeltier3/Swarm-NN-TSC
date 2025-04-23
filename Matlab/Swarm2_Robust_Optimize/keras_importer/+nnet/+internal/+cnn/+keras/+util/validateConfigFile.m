function Filename = validateConfigFile(Filename, isNetwork)
    % Copyright 2021 The MathWorks, Inc.
    if ~(isa(Filename,'char') || isa(Filename,'string'))
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FirstArgString')));
    end
    Filename = char(Filename);
    fileExists = exist(Filename, 'file');
    if ~fileExists
        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FileNotFound', Filename)));
    elseif fileExists == 7
        if isNetwork
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FileNotFoundButFolderExistsImportNetwork', Filename)));
        else
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:FileNotFoundButFolderExistsImportLayers', Filename)));
        end
    end
end
   