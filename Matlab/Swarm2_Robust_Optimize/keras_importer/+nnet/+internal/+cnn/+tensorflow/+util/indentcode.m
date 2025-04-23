function chars = indentcode(chars)
% Wrap a call to the internal indentcode function.
try
    chars = indentcode(chars);
catch
    % Do nothing if it failed.
end
end