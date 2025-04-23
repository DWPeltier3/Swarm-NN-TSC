function [labels, idx] = sortToDLTLabel(labels)
    % Copyright 2022 The MathWorks, Inc.
    
    labels = char(labels);
    numericalLabels = zeros(1, numel(labels)); 
    for i = 1:numel(labels)
        % SORT the incoming dimensions based on the SCBTU format
        switch labels(i)
            case 'S'
                numericalLabels(i) = 1; 
            case 'C'
                numericalLabels(i) = 2; 
            case 'B'
                numericalLabels(i) = 3; 
            case 'T'
                numericalLabels(i) = 4; 
            case 'U'
                numericalLabels(i) = 5; 
        end
    end

    % idx should be a stable ordering of the original data. 
    [~, idx] = sort(numericalLabels); 
    labels = labels(idx); 
end