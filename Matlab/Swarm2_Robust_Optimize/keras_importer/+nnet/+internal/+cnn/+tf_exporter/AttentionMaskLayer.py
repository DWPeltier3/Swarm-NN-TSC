class AttentionMaskLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(AttentionMaskLayer, self).__init__(name=name)

    def call(self, input):
        # input has shape [n t c]. Restructure input so that output has shape [n t t]
        shape = tf.shape(input)
        y = tf.gather(input, 0, axis = -1)
        y = tf.reshape(y, [shape[0], 1, shape[1]])
        return tf.repeat(y, shape[1], axis=1)
