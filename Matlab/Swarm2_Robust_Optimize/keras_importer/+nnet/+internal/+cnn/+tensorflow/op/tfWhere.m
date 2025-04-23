function y = tfWhere(condition)

%   Copyright 2020-2023 The MathWorks, Inc.

    conditionVal = condition.value;     
    conditionRank = condition.rank; 
         
    if isdlarray(conditionVal)
        conditionVal = extractdata(conditionVal); 
    end 
    
    I = cell(1, conditionRank);
    for i = 1:conditionRank
        I{i} = 0;
    end
    
    [I{:}] = ind2sub(size(conditionVal), find(conditionVal));
    yVal = horzcat(I{:});
    
    % Convert indices from 1-based to 0-based TF format.
    yVal = yVal - 1;
    yVal = permute(yVal, 2:-1:1);

    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', 2);
end 
