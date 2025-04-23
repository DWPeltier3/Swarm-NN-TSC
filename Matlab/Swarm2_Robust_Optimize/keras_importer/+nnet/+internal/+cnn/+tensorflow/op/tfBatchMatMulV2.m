function y = tfBatchMatMulV2(a, b)

%   Copyright 2021-2023 The MathWorks, Inc.

    aVal = a.value; 
    aRank = a.rank;
    
    bVal = b.value; 
    bRank = b.rank;
    
    % Thm: y' = (A * B)' == B' * A'. 
    % the input matrices are in reverse TF format now. (lets call them A',
    % and B'). Forward TF format is called A and B respectively. in TF, the last
    % two dimensions are multiplied. pagemtimes will multiply the first two
    % dimensions, Everything else is a page dimension. By the Thm above, we
    % will call pagemtimes(B', A'). This will return a transposed y. We then 
    % apply the reversed TF labels to get the correct MATLAB labels. If 
    % labels are unknown, we will output the reverse TF dimension tensor.
    yVal = pagemtimes(bVal, aVal);  
    yRank = ndims(yVal); 

    if aRank > bRank
        if yRank < aRank
            yRank = aRank;
        end
    else
        if yRank < bRank
            yRank = bRank;
        end
    end

    y = struct('value', yVal, 'rank', yRank); 
end 
