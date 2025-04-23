function tf = isInstalledIPT()

% Copyright 2022 The Mathworks, Inc.
tf = ~isempty(ver('images'));
end

