function y = tfCast(x, dstT)

%   Copyright 2020-2023 The MathWorks, Inc.
    xVal = x.value;
    xRank = x.rank;

    if isdlarray(xVal)
        xVal = extractdata(xVal); 
    end

    if isstruct(dstT)
        dstT = dstT.value;
    end

    switch dstT
        case {'DT_FLOAT', 'float32'}
            yVal = cast(single(xVal), 'like', xVal);
        case {'DT_DOUBLE', 'float64'}
            yVal = double(xVal);
        case {'DT_INT8', 'int8'}
            yVal = cast(int8(floor(xVal)), 'like', xVal); 
        case {'DT_INT16', 'int16'}
            yVal = cast(int16(floor(xVal)), 'like', xVal); 
        case {'DT_INT32', 'int32'}
            yVal = cast(int32(floor(xVal)), 'like', xVal); 
        case {'DT_INT64', 'int64'}
            yVal = cast(int64(floor(xVal)), 'like', xVal); 
        case {'DT_UINT8', 'uint8'}
            yVal = cast(uint8(floor(xVal)), 'like', xVal); 
        case {'DT_UINT16', 'uint16'}
            yVal = cast(uint16(floor(xVal)), 'like', xVal); 
        case {'DT_UINT32', 'uint32'}
            yVal = cast(uint32(floor(xVal)), 'like', xVal); 
        case {'DT_UINT64', 'uint64'}
            yVal = cast(uint64(floor(xVal)), 'like', xVal);
        case {'DT_BOOL', 'bool'}
            yVal = logical(xVal);
        otherwise
           assert(false, ['Unsupported Dtype ' dstT ' for tfCast'])
    end

    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', xRank);

end
