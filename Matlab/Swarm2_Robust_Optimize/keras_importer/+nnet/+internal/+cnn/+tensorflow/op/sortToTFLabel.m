function [X, Labels] = sortToTFLabel(X, Labels, channels_last)
% SORTTOTFLABEL Accepts an input X and Labels corresponding to the MATLAB
% conventions. It will return the expected TF dimension ordering. if
% channels_last is set to false, the channels first convention is followed
% by setting the labels to NCHW format.

%   Copyright 2020-2022 The MathWorks, Inc.

% Example: 
% myDlarray = dlarray(rand(32, 32, 3), "SSCB");
% [permutationVec, tfdims] = sortToTFLabel(1:ndims(myDlarray), dims(myDlarray)) 

% permutationVec =
%      4     1     2     3
% tfdims =
%     'BSSC'

if nargin < 3
    channels_last = true; 
end

Labels = char(Labels);
numericalLabels = zeros(1, numel(Labels)); 
if channels_last
    for i = 1:numel(Labels)
        % SORT the incoming dimensions based on the BU*TS*C
        % channels_last format.
        switch Labels(i)
            case 'B'
                numericalLabels(i) = 1; 
            case 'U'
                numericalLabels(i) = 2; 
            case 'T'
                numericalLabels(i) = 3; 
            case 'S'
                numericalLabels(i) = 4; 
            case 'C'
                numericalLabels(i) = 5; 
        end
    end
else
    for i = 1:numel(Labels)
        % SORT the incoming dimensions based on the BU*TCS*
        % channels_first format.
        switch Labels(i)
            case 'B'
                numericalLabels(i) = 1; 
            case 'U'
                numericalLabels(i) = 2; 
            case 'T'
                numericalLabels(i) = 3; 
            case 'C'
                numericalLabels(i) = 4; 
            case 'S'
                numericalLabels(i) = 5; 
        end
    end
end

% idx should be a stable ordering of the original data. 
[~, idx] = sort(numericalLabels); 
if numel(idx) > numel(X)
    % DLT has added an extra dimension
    X = 1:numel(idx); 
end
X = X(idx);
Labels = Labels(idx); 
end
   
