function setAdditionalResourceLocation()

% Copyright 2018 The MathWorks, Inc.

m = message('nnet_cnn_kerasimporter:keras_importer:ConfigJSONButNoWeightfile'); 
try  
        m.getString(); 
catch 
        matlab.internal.msgcat.setAdditionalResourceLocation(supportPackageRootDir()) 
end
end

function rootDir = supportPackageRootDir()
%Return parent directory of the support package at runtime

% To get to the matlab root, we need to remove the trailing 9 parts of the
% fullpath:
rootDir = mfilename('fullpath');
for i = 1:9
    rootDir = fileparts(rootDir);
end
end
