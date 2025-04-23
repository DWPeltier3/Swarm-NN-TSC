function [topK, indices] = tfTopKV2(x, k)

%   Copyright 2022-2023 The MathWorks, Inc.

    xVal = x.value;
    xRank = x.rank;

    % If k is a struct extract the numeric value
    if isstruct(k)
        kVal = k.value;
        if isdlarray(kVal)
            kVal = extractdata(kVal);
        end
    else
        % if k is numeric
        kVal = k;
    end
    
    % maxk is not a dlarray method
    xVal = extractdata(xVal);
    [topKVal, idxVal] = maxk(xVal, kVal, 1);
    
    % convert indices to 0-based-indexing
    idxVal = idxVal - 1;

    topKVal = dlarray(topKVal);
    idxVal = dlarray(idxVal);

    % assign output rank:
    topK = struct('value', topKVal, 'rank', xRank);
    indices = struct('value', idxVal, 'rank', xRank);
end
