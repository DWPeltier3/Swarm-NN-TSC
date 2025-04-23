function maybeKerasFormat = checkKerasFileFormat(FilePath)
    % Copyright 2021 The MathWorks, Inc.  
    [~,~,ext] = fileparts(FilePath);
    switch lower(ext)
        case {'.h5','.hdf5', '.json'}
            maybeKerasFormat = true;
        otherwise
            maybeKerasFormat = false;
    end
end