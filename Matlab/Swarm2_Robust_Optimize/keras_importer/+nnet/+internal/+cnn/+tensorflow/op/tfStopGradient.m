function y = tfStopGradient(input)

%   Copyright 2020-2023 The MathWorks, Inc.

    if isstruct(input)
        yRank = input.rank; 
        input = input.value; 
        if isdlarray(input)
            % z continues as a copy of the original input. But the path to the
            % original is maintained via a zero gradient skip connection. 
            y = dlarray(input.extractdata) + (0 .* input); 
        else
            y = input; 
        end 
        y = struct('value', y, 'rank', yRank);
    else
        y = input;
    end
end 
