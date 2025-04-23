class RescaleSymmetricLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(RescaleSymmetricLayer, self).__init__(name=name)
        self.min = tf.Variable(initial_value=tf.zeros(shape), trainable=False)
        self.max = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        # Scale the range [min, Max] into [-1, 1]
        return (input - self.min)/(self.max - self.min) * 2 - 1
