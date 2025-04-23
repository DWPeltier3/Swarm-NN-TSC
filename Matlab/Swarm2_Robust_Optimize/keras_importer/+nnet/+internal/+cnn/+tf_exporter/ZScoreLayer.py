class ZScoreLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(ZScoreLayer, self).__init__(name=name)
        self.mean = tf.Variable(initial_value=tf.zeros(shape), trainable=False)
        self.std = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        # Compute z-score of input
        return (input - self.mean)/self.std
