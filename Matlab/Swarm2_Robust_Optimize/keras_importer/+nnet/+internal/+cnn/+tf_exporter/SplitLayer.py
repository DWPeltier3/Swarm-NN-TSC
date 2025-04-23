class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, num_or_size_splits, axis=0, num=None, name=None):
        # A layer wrapper for tf.split
        super(SplitLayer, self).__init__(name=name)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.num = num

    def call(self, input):
        # Pass the arguments along to tf.split
        return tf.split(input, self.num_or_size_splits, self.axis, self.num)