function Conn = renameConn(Conn, NameTable)
% Apply the NameTable to this Connection. Can be an input or output
% connection.
if ~isempty(NameTable)
    Idx = find(strcmp(Conn.FromLayerName, NameTable.FromName) & (NameTable.FromNum == Conn.FromOutputNum));
    assert(numel(Idx) < 2); % At most one rule should apply at a time.
    if ~isempty(Idx)
        % Apply the rule and recurse
        Conn.FromLayerName = NameTable.ToName{Idx};
        Conn.FromOutputNum = NameTable.ToNum(Idx);
        Conn = nnet.internal.cnn.keras.util.renameConn(Conn, NameTable);
    end
end
end