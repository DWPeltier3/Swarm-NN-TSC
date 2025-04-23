class ChannelShufflingLayer(tf.keras.layers.Layer):
    def __init__(self, groups, name=None):
        super(ChannelShufflingLayer, self).__init__(name=name)
        self.groups = tf.cast(groups, tf.int32)

    def call(self, input):
        # input has shape [n h w c]. Split the c dimension, permute, 
        # then rejoin:
        shape = tf.shape(input);
        n = shape[0]
        h = shape[1]
        w = shape[2]
        Z = tf.reshape(input, [n, h, w, self.groups, -1])
        Z = tf.transpose(Z, [0,1,2,4,3])
        Z = tf.reshape(Z, shape)
        return Z
