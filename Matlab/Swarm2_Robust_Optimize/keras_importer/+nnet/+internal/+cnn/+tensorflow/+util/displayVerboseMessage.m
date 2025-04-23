function displayVerboseMessage(MsgId, VerboseFlag)
    % Copyright 2021 The MathWorks, Inc.  
    if VerboseFlag
        msg = message(MsgId);
        disp(getString(msg));
    end
end