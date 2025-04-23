function MATLABNames = iMakeLegalMATLABNames(TFNames)
%

%   Copyright 2020-2021 The MathWorks, Inc.

import nnet.internal.cnn.keras.util.*;
if isempty(TFNames)
    MATLABNames = TFNames;
    return;
end

MaxVariableNameLength = 60;
% Replace reserved words
MATLABNames = replaceReservedWords(TFNames);
% Replace slashes and colons with different strings
MATLABNames = strrep(MATLABNames, '/', '_');
MATLABNames = convertToCamelCase(MATLABNames); 
MATLABNames = strrep(MATLABNames, ':', '_c_');
MATLABNames = removeCommonPatterns(MATLABNames);
% Make them shorter, maintaining uniqueness
MATLABNames = matlab.lang.makeUniqueStrings(MATLABNames, {}, MaxVariableNameLength);
% Make them matlab-legal
MATLABNames = matlab.lang.makeValidName(MATLABNames);
if numel(MATLABNames) ~= numel(unique(MATLABNames))
    warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnableToGenerateLegalVariableOrFunctionName');
end
end

function Names = replaceReservedWords(Names)
ReservedWords = reservedWords();
for i=1:numel(Names)
    if ismember(Names{i}, ReservedWords)
        Names{i} = [Names{i} 'Var'];
    end
end
end

function Names = reservedWords()

% Making the reserved names a persistent variable for optimizing performance 
persistent reservedNames
if isempty(reservedNames)
    reservedNames = methods('dlarray');
    reservedNames = [reservedNames ; {'ones', 'complex', 'eye', 'false', 'true', 'rand', 'randn', 'zeros'}'];
end
Names = reservedNames;

end

function Names = convertToCamelCase(Names)
    %Converts pythonic snake_case to matlaby camelCase
    for i=1:numel(Names)
        % locate all underscores (with exception of underscore at the end).
        underscores = find(Names{i}(1:end-1) == '_');
        
        % Make the next character uppercase.
        underscores = underscores + 1;
        Names{i}(underscores) = upper(Names{i}(underscores));
        
        % Remove all underscores
        Names{i} = strrep(Names{i}, '_', '');
    end
end 

function Names = removeCommonPatterns(Names)
CommonPatterns = commonPatterns();
for i=1:numel(Names)
    for j = 1:numel(CommonPatterns)
       Names{i} = strrep(Names{i}, CommonPatterns{j}, '');
    end 
end

end 

function Names = commonPatterns()
Names = {'Inference', 'CallAndReturn', 'ConditionalLosses'}; 
end 
