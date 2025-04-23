function y = tfMatMul(a, b, transpA, transpB)

%   Copyright 2020-2023 The MathWorks, Inc.

    aVal = a.value;
    aRank = a.rank;

    bVal = b.value;
    bRank = b.rank;

    tA = 'none';
    tB = 'none';

    if aRank ~= 2 || bRank ~= 2
        error('MatMul is only supported for input tensors having a rank of 2.');
    end
    
    if transpA
        tA = 'transpose';
    end
    
    if transpB
        tB = 'transpose';
    end

    yVal = pagemtimes(bVal, tB, aVal, tA);  
    yRank = ndims(yVal); 

    y = struct('value', yVal, 'rank', yRank); 
end 
