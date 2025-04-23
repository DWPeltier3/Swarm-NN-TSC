function rootDir = supportPackageRootDir()

    %   Copyright 2022 The MathWorks, Inc.

%Return parent directory of the support package at runtime
% To get to the matlab root, we need to remove the trailing 9 parts of the
% fullpath:
rootDir = mfilename('fullpath');
for i = 1:9
    rootDir = fileparts(rootDir);
end
end
