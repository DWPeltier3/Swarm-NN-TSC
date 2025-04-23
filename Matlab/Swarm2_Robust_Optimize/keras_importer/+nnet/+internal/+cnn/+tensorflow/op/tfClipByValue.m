function y = tfClipByValue(t, clipValueMin, clipValueMax)
    %{{import_statement}}

%   Copyright 2023 The MathWorks, Inc.

    tVal = t.value; 
    tRank = t.rank;

    if isstruct(clipValueMin)
        clipValueMin = clipValueMin.value;
    end	 
    
    if isstruct(clipValueMax)
        clipValueMax = clipValueMax.value;
    end
	
    yVal = min(max(tVal, clipValueMin), clipValueMax);
    y = struct('value', yVal, 'rank', tRank);
end 
