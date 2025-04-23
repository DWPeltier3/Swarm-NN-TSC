classdef Tensor < handle

% Copyright 2017 The Mathworks, Inc.

    properties
        FromLayerName
        FromOutputNum
    end
    
    methods
        function this = Tensor(FromLayerName, FromOutputNum)
            this.FromLayerName = FromLayerName;
            this.FromOutputNum = FromOutputNum;
        end
    end
end
