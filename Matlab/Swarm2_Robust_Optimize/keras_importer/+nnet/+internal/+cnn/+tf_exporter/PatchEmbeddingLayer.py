class PatchEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, kernel_shape=None, bias_shape=None, flip=None, temporal=None, name=None):
        super(PatchEmbeddingLayer, self).__init__(name=name)
        self.patch_size = patch_size
        # Learnable parameters: These have been exported from MATLAB and will be loaded automatically from the weight file:
        self.kernel = tf.Variable(name="kernel", initial_value=tf.zeros(kernel_shape), trainable=True)
        self.bias = tf.Variable(name="bias", initial_value=tf.zeros(bias_shape), trainable=True)
        self.flip = flip
        self.temporal = temporal

    def call(self, inputs):
        
        outputs = tf.nn.convolution(
            inputs,
            self.kernel,
            strides=self.patch_size,
            padding='VALID')
        
        outputs = tf.nn.bias_add(outputs, self.bias)

        # Flatten the spatial dimensions in outputs
        outputs_shape = tf.shape(outputs)
        len_outputs_shape = len(outputs_shape)
        if self.temporal:
            if self.flip:
                # Swap the spatial dimensions first:
                # outputs format: BTSSC (for 2d)
                # flip_dims: [3, 2]
                # perm: [0, 1, 3, 2, 4]
                flip_dims = tf.range(len_outputs_shape-2, 1, -1)
                outputs = tf.transpose(outputs, perm=tf.concat([tf.constant([0,1]), flip_dims, tf.expand_dims(tf.constant(len_outputs_shape-1),0)],0))

            flatten_val = tf.math.reduce_prod(outputs_shape[2:len_outputs_shape-1])
            outputs = tf.reshape(outputs, shape=tf.stack([tf.constant(-1), outputs_shape[1], flatten_val, outputs_shape[-1]]))
        else:
            if self.flip:
                # Swap the spatial dimensions first:
                # outputs format: BSSC (for 2d)
                # flip_dims: [2, 1]
                # perm: [0, 2, 1, 3]
                flip_dims = tf.range(len_outputs_shape-2, 0, -1)
                outputs = tf.transpose(outputs, perm=tf.concat([tf.expand_dims(tf.constant(0),0), flip_dims, tf.expand_dims(tf.constant(len_outputs_shape-1),0)],0))

            flatten_val = tf.math.reduce_prod(outputs_shape[1:len_outputs_shape-1])
            outputs = tf.reshape(outputs, shape=tf.stack([tf.constant(-1), flatten_val, outputs_shape[-1]]))

        return outputs