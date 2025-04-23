class BroadcastLayer(tf.keras.layers.Layer):
    def __init__(self, BT_length, observation_shape, name=None):
        # BT_length is the number of B and T dimensions of the input tensor. 
        # For example, BTSSC has length 2 while BSSC has length 1.
        # observation_shape is the target shape of the rest of the tensor,
        # in these examples, the SSC part.
        super(BroadcastLayer, self).__init__(name=name)
        self.BT_length = BT_length
        self.observation_shape = tf.constant(observation_shape)

    def call(self, input):
        # Concat the input's BT shape to observation_shape, then broadcast 
        # the input to that shape.
        BTshape = tf.shape(input)[0:self.BT_length]
        toShape = tf.concat([BTshape, self.observation_shape], 0)
        return tf.broadcast_to(input, toShape)