function y = tfL2Loss(t)

%   Copyright 2020-2023 The MathWorks, Inc.

    y.value = dlarray(sum(t.value .^ 2, 'all') ./ 2);
    y.rank = 0; 
end
