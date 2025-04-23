function [selectedIndices, selectedScores, validOutputs] = tfNonMaxSuppressionV5(boxes, scores, maxOutputSize, ...
    iouThreshold, scoreThreshold, softNmsSigma)

% Copyright 2022-2023 The MathWorks, Inc.
% Implements the TF NonMaxSuppression operator: https://www.tensorflow.org/api_docs/python/tf/raw_ops/NonMaxSuppressionV5

boxesVal = boxes.value; % TF:a 2-D tensor of shape [num_boxes, 4].
scoresVal = scores.value; % TF:a 1-D tensor/vector of shape [num_boxes].
maxOutputSizeVal = maxOutputSize.value;
iouThresholdVal = iouThreshold.value;
scoreThresholdVal = scoreThreshold.value;
softNmsSigmaVal = softNmsSigma.value;

if(softNmsSigmaVal > 0)
    error('NonMaxSuppression does not support soft_nms_sigma greater than 0.'); 
end

if isempty(boxesVal)
    selectedIndices.value = dlarray([]);
    selectedIndices.rank = 1;
    selectedScores.value = dlarray([]);
    selectedScores.rank = 1;
    validOutputs.value = 0;
    validOutputs.rank = 0;
    return
end

% Extract data from input dlarray objects.
boxesVal = iExtractData(boxesVal);  % MATLAB:a 2-D dlarray of shape [4 num_boxes].
boxesVal = boxesVal'; % [num_boxes 4]
scoresVal = iExtractData(scoresVal); % MATLAB:a 2-D dlarray of shape [1 num_boxes].

maxOutputSizeVal = iExtractData(maxOutputSizeVal);
iouThresholdVal = iExtractData(iouThresholdVal);
scoreThresholdVal = iExtractData(scoreThresholdVal);

[numBoxes, ~] = size(boxesVal);

% Perform NMS across batches.
% TF box format is [y1 x1 y2 x2] in spatial coordinate and defines
% diagonal pairs. The order of the pairs is not defined. Compute the
% min and max coordinates.
xmin = min(boxesVal(:,[2 4]),[],2);
xmax = max(boxesVal(:,[2 4]),[],2);
ymin = min(boxesVal(:,[1 3]),[],2);
ymax = max(boxesVal(:,[1 3]),[],2);

% Convert min and max coordinates to [x y w h].
bboxes = [xmin ymin xmax-xmin ymax-ymin];

 % Keep boxes above score threshold.
keep = scoresVal > scoreThresholdVal;
boxesAfterScoreThreshold = bboxes(keep,:);
scoresAfterScoreThreshold = scoresVal(keep,:);

% Create the index list of boxes 
idx = (1:numBoxes)';

% Track original indices of boxes that were kept.
idx1 = idx(keep);

% selectStrongestBbox only supports valid box inputs with width and
% height > 0. However, TF NMS allows these types of boxes and it
% does not suppress them in its output. Here, we remove invalid
% boxes, apply selectStrongestBbox, and then add the invalid boxes
% back to mimic TF NMS output.
%
% Select valid boxes and keep track of valid box indices.
valid = all(boxesAfterScoreThreshold(:,[3 4]) > 0, 2);
validInd = find(valid);
validBoxes = boxesAfterScoreThreshold(valid,:);
validScores = scoresAfterScoreThreshold(valid,:);

if ~isempty(validBoxes)
    [~,~,index] = selectStrongestBbox(validBoxes,validScores,...
            'RatioType','Union',...
            'OverlapThreshold',iouThresholdVal,...
            'NumStrongest',maxOutputSizeVal);
end

index = validInd(index);

% Reorder indices by score.
[~, ord] = sort(scoresAfterScoreThreshold(index),'descend');
index = index(ord);

% Get index into original input boxes before score threshold.
index = idx1(index);

selectedIndices.value = dlarray(index - 1);
selectedIndices.rank = 1;
selectedScores.value = dlarray(scoresVal(index));
selectedScores.rank = 1;
validOutputs.value = dlarray(numel(index));
validOutputs.rank = 0;

function x = iExtractData(x)
    if isdlarray(x)
        x = extractdata(x);
    end
end

end
