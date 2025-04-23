function path = validatePackageName(path)
    % Copyright 2022 The MathWorks, Inc.  

    path = char(path);

    if ~isWritable
        nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:FolderNotWritable', pwd);
    end

end
function writeStatus = isWritable
% Checks if current directory is writable
    tempDir = strsplit(tempname, filesep);
    tempDir = tempDir{end};
    [writeStatus, ~] = mkdir(tempDir); 
    if writeStatus
        rmdir(tempDir);
    end
end