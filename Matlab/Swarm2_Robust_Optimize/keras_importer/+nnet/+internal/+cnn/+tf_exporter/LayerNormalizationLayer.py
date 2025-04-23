class LayerNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, axis, epsilon, offset_shape=None, scale_shape=None, name=None):
        super(LayerNormalizationLayer, self).__init__(name=name)
        self.epsilon = epsilon
        self.axis = axis
        # Learnable parameters: These have been exported from MATLAB and will be loaded automatically from the weight file:
        self.gamma = tf.Variable(name="gamma", initial_value=tf.zeros(scale_shape), trainable=True)
        self.beta = tf.Variable(name="beta", initial_value=tf.zeros(offset_shape), trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, self.axis, keepdims=True)

        outputs = tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset=self.beta,
                scale=self.gamma,
                variance_epsilon=self.epsilon)

        return outputs
