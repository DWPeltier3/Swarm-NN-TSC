class EmbeddingConcatenationLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_shape=None, name=None):
        super(EmbeddingConcatenationLayer, self).__init__(name=name)
        # Learnable parameters: These have been exported from MATLAB and will be loaded automatically from the weight file:
        self.kernel = tf.Variable(name="kernel", initial_value=tf.zeros(kernel_shape), trainable=True)

    def call(self, inputs):

        embed_vec = self.kernel
        input_shape = tf.shape(inputs)
        
        if len(input_shape) > 2:
            embed_vec = tf.expand_dims(embed_vec, axis=0)
            embed_vec = tf.tile(embed_vec, tf.stack([input_shape[0], tf.constant(1), tf.constant(1)]))

        return tf.concat([embed_vec, inputs], axis=-2)