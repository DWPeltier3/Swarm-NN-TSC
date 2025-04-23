function rootDir = getKerasImporterRoot()
%Return parent directory of Keras importer support package at runtime

% Copyright 2017 The MathWorks, Inc.

% To get to the matlab root, we need to remove the trailing 9 parts of the
% fullpath:
rootDir = mfilename('fullpath');
for i = 1:9
    rootDir = fileparts(rootDir);
end
end