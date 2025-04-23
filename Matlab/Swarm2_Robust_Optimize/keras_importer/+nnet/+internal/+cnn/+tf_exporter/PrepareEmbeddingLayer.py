class PrepareEmbeddingLayer(tf.keras.layers.Layer):
    # vocab_size is the number of words + 1 when accept_oov is true,
    # to implement MATLAB's appended "other" vocab item.
    def __init__(self, vocab_size, do_squeeze, accept_oov, name=None):
        super(PrepareEmbeddingLayer, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.do_squeeze = do_squeeze
        self.accept_oov = accept_oov

    def call(self, input):
        # Input format is either BTC=BT1, or BT. Output is BT, the required
        # input shape for the Embedding layer.
        # Map 0 and out of bounds indices to vocab_size if 
        # accept_oov is true, then subtract 1 to make them origin-0:
        output = input
        if self.accept_oov:
            output = tf.where(tf.equal(output, 0.), self.vocab_size, output)
            output = tf.where(output > self.vocab_size, self.vocab_size, output)
        output = output-1
        # Remove the trailing singleton third axis if present
        if self.do_squeeze:
            output = tf.squeeze(output, axis=2)
        return output