function y = tfTranspose(x, perm)

%   Copyright 2021-2024 The MathWorks, Inc.

    yRank = x.rank;
    
    % transpose can only happen for xrank > 1
    if yRank < 2
        y = x;
        return;
    end
    
    xVal = x.value;
    if isstruct(perm)
        perm = perm.value;
    end
    
    % perm will always be rank-1 and numel(perm) should equal xrank
    % convert TensorFlow permutation vector to 1-indexed reverse TensorFlow
    % indexing
    perm = yRank - flip(perm);
    
    y = permute(xVal, perm);
    
    y = struct('value', y, 'rank', yRank); 
end
