class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, axis, params_shape=None, name=None):
        super(PositionEmbeddingLayer, self).__init__(name=name)
        self.axis = axis
        # Learnable parameter: This has been exported from MATLAB and will be loaded automatically from the weight file:
        self.params = tf.Variable(name="params", initial_value=tf.zeros(params_shape), trainable=True)

    def call(self, inputs):

        inputs_shape = tf.shape(inputs)
        position_axis = self.axis

        positions = tf.range(inputs_shape[position_axis])
        embedding = tf.gather(self.params, positions)

        if position_axis == 0:
            outputs = embedding
        else:
            embedding = tf.expand_dims(embedding, axis=0)
            len_inputs_shape = len(inputs_shape)
            if len_inputs_shape < 4:
                # Input shape is of the for BTC or BSC
                outputs = tf.tile(embedding, multiples=[inputs_shape[0],1,1])
            elif position_axis == 1:
                # Input shape is of the form BTS...C
                embedding = tf.expand_dims(embedding, axis=2)
                outputs = tf.tile(embedding, multiples=[inputs_shape[0],1,inputs_shape[2],1])
                if len_inputs_shape > 4: # Expand output along other S dimensions
                    for i in range(3, len_inputs_shape-1):
                        outputs = tf.expand_dims(outputs, axis=i)
                        multiples = tf.concat([tf.ones([i,],tf.int32), tf.expand_dims(inputs_shape[i],0), tf.expand_dims(tf.constant(1),0)], 0)
                        outputs = tf.tile(outputs, multiples)
            else: # Input shape is of the form BTSC
                embedding = tf.expand_dims(embedding, axis=1)
                outputs = tf.tile(embedding, multiples=[inputs_shape[0],inputs_shape[1],1,1])
        return outputs