function Conn = connFromTensorSpec(Spec)
% Spec is a KerasDAGInputTensorSpec or KerasDAGOutputTensorSpec
Conn = nnet.internal.cnn.keras.Tensor(Spec.LayerName, Spec.LayerOutputNum);
end
