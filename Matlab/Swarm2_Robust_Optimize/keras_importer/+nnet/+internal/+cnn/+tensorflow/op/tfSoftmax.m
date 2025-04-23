function y = tfSoftmax(logits)

%   Copyright 2022-2023 The MathWorks, Inc.
%   Computes softmax activations.
%
%   logits: A Tensor. 
%          Must be one of the following types: half, bfloat16, float32, float64. 
%          2-D with shape [batch_size, num_classes].    
%
logitsRank = logits.rank;
logitsVal = logits.value;

% Input verification
if logitsRank <= 1
    error('Softmax is only supported for input tensors having a rank of 2 or more.');
end

midSLabels = char('S' + zeros(1, logitsRank - 2));
TFXLabels = ['B' midSLabels 'C'];

% Permute to forward tensorflow format and apply labels
logitsVal = permute(stripdims(logitsVal), logitsRank:-1:1);
    
yVal = softmax(logitsVal, "DataFormat", TFXLabels);
yVal = permute(yVal, logitsRank:-1:1);

% assign output rank:
y = struct('value', yVal, 'rank', logitsRank);
end