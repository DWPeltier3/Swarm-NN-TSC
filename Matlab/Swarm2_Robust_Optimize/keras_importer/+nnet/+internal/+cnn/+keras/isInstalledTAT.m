function tf = isInstalledTAT()

% Copyright 2018 The Mathworks, Inc.
tf = ~isempty(ver('TextAnalytics'));
end