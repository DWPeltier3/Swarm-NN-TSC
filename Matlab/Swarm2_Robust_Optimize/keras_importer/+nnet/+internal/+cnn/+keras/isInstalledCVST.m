function tf = isInstalledCVST()

% Copyright 2018 The Mathworks, Inc.
tf = ~isempty(ver('vision'));
end