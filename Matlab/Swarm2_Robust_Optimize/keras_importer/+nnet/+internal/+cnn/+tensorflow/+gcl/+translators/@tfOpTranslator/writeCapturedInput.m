function code = writeCapturedInput(this, node_def, MATLABOutputName, TFConstants) 
%

%   Copyright 2022 The MathWorks, Inc.

    if ~isempty(node_def.attr.value.tensor.tensor_shape.dim)
        [tensorndims, ~] = size(node_def.attr.value.tensor.tensor_shape.dim);
    else
        tensorndims = 0;
    end
    tensor_shape = iGetTFShape(node_def.attr.value.tensor.tensor_shape); 
    switch node_def.attr.dtype.type
        case 'DT_FLOAT'
            if ~isempty(node_def.attr.value.tensor.float_val) && prod(tensor_shape) == 1 
                f = single(node_def.attr.value.tensor.float_val);
                code = MATLABOutputName + " = single(" + f + ")"; 
            else
                % We have a vector of data here 
                if ~isempty(node_def.attr.value.tensor.float_val)
                    f = node_def.attr.value.tensor.float_val*ones(prod(tensor_shape), 1, 'single'); 
                else
                    f = typecast(matlab.net.base64decode(node_def.attr.value.tensor.tensor_content), 'single');
                end 
                shape = {node_def.attr.value.tensor.tensor_shape.dim.size}; 
                shape = str2double(shape);
                if numel(shape) == 1
                    shape = [shape 1]; 
                end 
                f = reshape(f, shape);
                if ~node_def.attr.value.tensor.tensor_shape.unknown_rank
                    numDimsF = numel(node_def.attr.value.tensor.tensor_shape.dim); 
                    f = nnet.internal.cnn.tensorflow.util.rowMajorToColumnMajor(f, numDimsF); 
                end
                constName = TFConstants.updateConstant(MATLABOutputName, f, tensorndims); 
                code = MATLABOutputName + " = " + this.LAYERREF + "." + constName; 
            end
            
           case 'DT_DOUBLE'
            if ~isempty(node_def.attr.value.tensor.double_val) && prod(tensor_shape) == 1 
                f = double(node_def.attr.value.tensor.double_val);
                code = MATLABOutputName + " = double(" + num2str(f) + ")"; 
            else
                % We have a vector of data here 
                if ~isempty(node_def.attr.value.tensor.double_val)
                    f = node_def.attr.value.tensor.double_val*ones(prod(tensor_shape), 1); 
                else
                    f = typecast(matlab.net.base64decode(node_def.attr.value.tensor.tensor_content), 'double');
                end 
                shape = {node_def.attr.value.tensor.tensor_shape.dim.size}; 
                shape = str2double(shape);
                if numel(shape) == 1
                    shape = [shape 1]; 
                end 
                f = reshape(f, shape);
                if ~node_def.attr.value.tensor.tensor_shape.unknown_rank
                    numDimsF = numel(node_def.attr.value.tensor.tensor_shape.dim); 
                    f = nnet.internal.cnn.tensorflow.util.rowMajorToColumnMajor(f, numDimsF); 
                end
                constName = TFConstants.updateConstant(MATLABOutputName, f, tensorndims); 
                code = MATLABOutputName + " = " + this.LAYERREF + "." + constName;
            end
            
        case 'DT_INT32'
            if ~isempty(node_def.attr.value.tensor.int_val) && prod(tensor_shape) == 1 
                f = int32(node_def.attr.value.tensor.int_val);
                code = MATLABOutputName + " = double(" + num2str(f) + ")"; 
            else 
                % We have a vector of data here 
                if ~isempty(node_def.attr.value.tensor.int_val)
                    f = node_def.attr.value.tensor.int_val*ones(prod(tensor_shape), 1); 
                else
                    f = typecast(matlab.net.base64decode(node_def.attr.value.tensor.tensor_content), 'int32');
                    f = double(f);
                end
                shape = {node_def.attr.value.tensor.tensor_shape.dim.size}; 
                shape = str2double(shape); 
                if numel(shape) == 1
                    shape = [shape 1]; 
                end 
                f = reshape(f, shape); 
                if ~node_def.attr.value.tensor.tensor_shape.unknown_rank
                    numDimsF = numel(node_def.attr.value.tensor.tensor_shape.dim); 
                    f = nnet.internal.cnn.tensorflow.util.rowMajorToColumnMajor(f, numDimsF); 
                end 
                constName = TFConstants.updateConstant(MATLABOutputName, f, tensorndims); 
                code = MATLABOutputName + " = " + this.LAYERREF + "." + constName; 
            end

        case 'DT_INT64'
            if ~isempty(node_def.attr.value.tensor.int64_val) && prod(tensor_shape) == 1 
                f = int64(str2double(node_def.attr.value.tensor.int64_val{1}));
                code = MATLABOutputName + " = double(" + num2str(f) + ")"; 
            else 
                % We have a vector of data here 
                if ~isempty(node_def.attr.value.tensor.int_val)
                    f = node_def.attr.value.tensor.int_val*ones(prod(tensor_shape), 1); 
                else
                    f = typecast(matlab.net.base64decode(node_def.attr.value.tensor.tensor_content), 'int64');
                    f = double(f);
                end
                shape = {node_def.attr.value.tensor.tensor_shape.dim.size}; 
                shape = str2double(shape); 
                if numel(shape) == 1
                    shape = [shape 1]; 
                end 
                f = reshape(f, shape); 
                if ~node_def.attr.value.tensor.tensor_shape.unknown_rank
                    numDimsF = numel(node_def.attr.value.tensor.tensor_shape.dim); 
                    f = nnet.internal.cnn.tensorflow.util.rowMajorToColumnMajor(f, numDimsF); 
                end 
                constName = TFConstants.updateConstant(MATLABOutputName, f, tensorndims); 
                code = MATLABOutputName + " = " + this.LAYERREF + "." + constName; 
            end

        case 'DT_STRING' 
            if ~isempty(node_def.attr.value.tensor.string_val)
                f = char(matlab.net.base64decode(node_def.attr.value.tensor.string_val{1})); 
            elseif ~isempty(node_def.attr.value.tensor.tensor_content)
                f = typecast(matlab.net.base64decode(node_def.attr.value.tensor.tensor_content), 'char');
            else 
                f = string.empty;
            end 
            % Assume tensordim = 1 for strings
            constName = TFConstants.updateConstant(MATLABOutputName, f, 1); 

            code = MATLABOutputName + " = " + this.LAYERREF + "." + constName;

        case 'DT_BOOL'
            if ~isempty(node_def.attr.value.tensor.bool_val) && prod(tensor_shape) == 1 
                f = logical(node_def.attr.value.tensor.bool_val);
                if f
                    code = MATLABOutputName + " = true"; 
                else 
                    code = MATLABOutputName + " = false"; 
                end 
            else 
                % We have a vector of data here
                if ~isempty(node_def.attr.value.tensor.bool_val)
                    val = logical(node_def.attr.value.tensor.bool_val); 
                    if val
                        f = true(prod(tensor_shape), 1);
                    else
                        f = false(prod(tensor_shape), 1);
                    end
                else
                    f = logical(matlab.net.base64decode(node_def.attr.value.tensor.tensor_content));
                end
                shape = {node_def.attr.value.tensor.tensor_shape.dim.size}; 
                shape = str2double(shape);
                if numel(shape) == 1
                    shape = [shape 1]; 
                end 
                f = reshape(f, shape); 
                if ~node_def.attr.value.tensor.tensor_shape.unknown_rank
                    numDimsF = numel(node_def.attr.value.tensor.tensor_shape.dim); 
                    f = nnet.internal.cnn.tensorflow.util.rowMajorToColumnMajor(f, numDimsF); 
                end 
                constName = TFConstants.updateConstant(MATLABOutputName, f, tensorndims); 
                code = MATLABOutputName + " = " + this.LAYERREF + "." + constName; 
            end
        otherwise
            code = "% unrecognized datatype " + node_def.attr.dtype.type;
    end
    code = code + ";" + newline;
end

function shapeVec = iGetTFShape(tensor_shape)
    if isempty(tensor_shape.dim)
        shapeVec = [];
    else 
        shapeVec = cellfun(@(x)(str2num(x)), {tensor_shape.dim.size}); 
    end
end 
