function [outputCode] = processFunctionOutputs(this, numOutputs, MATLABOutputName)
   %   Copyright 2023 The MathWorks, Inc.

% Generate code for post-processing and labeling outputs of the OpLambda function layer
    outputCode = "";
    
    if numOutputs > 1
        % Multiple outputs, hence, output values captured in a cell array of structs
        for i = 1:numOutputs
            outputCode = outputCode + MATLABOutputName + "{" + num2str(i) + "}" + ...
                                    " = addOutputLabel(" + MATLABOutputName + "{" + num2str(i) + "}" + ", "+ num2str(i) + ", " + this.LAYERREF + ");" + newline;
            outputCode = outputCode + "varargout{" + num2str(i) + "} = iPermuteToForwardTF(" + MATLABOutputName + "{" + num2str(i) + "}" + ...
                                    ".value," + MATLABOutputName  + "{" + num2str(i) + "}" + "." + this.RANKFIELDNAME + ");" + newline;
        end
    else
        % Single output, hence, output value captured in a single struct variable
        outputCode = outputCode + MATLABOutputName + " = addOutputLabel(" + MATLABOutputName + ", 1, " + this.LAYERREF + ");" + newline;
        outputCode = outputCode + "varargout{1} = iPermuteToForwardTF(" + MATLABOutputName + ".value," + MATLABOutputName + "." + this.RANKFIELDNAME + ");" + newline;
    end
end

