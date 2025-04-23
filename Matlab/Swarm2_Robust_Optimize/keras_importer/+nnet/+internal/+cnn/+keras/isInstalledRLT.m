function tf = isInstalledRLT()

% Copyright 2023 The Mathworks, Inc.
tf = ~isempty(ver('rl'));
end