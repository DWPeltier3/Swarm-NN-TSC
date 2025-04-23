class SequenceUnfoldingLayer(tf.keras.layers.Layer):
    def __init__(self, dataShape, name=None):
        super(SequenceUnfoldingLayer, self).__init__(name=name)
        self.dataShape = dataShape;

    def call(self, X, batchSize):
        return tf.reshape(X, (batchSize, -1) + self.dataShape)
