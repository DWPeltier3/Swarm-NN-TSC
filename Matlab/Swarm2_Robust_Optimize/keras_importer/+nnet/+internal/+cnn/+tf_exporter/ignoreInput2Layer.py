class ignoreInput2Layer(tf.keras.layers.Layer):
    # This layer is placed before some Resizing layers. It takes 2 inputs 
    # and passes the first one through.
    def __init__(self, name=None):
        super(ignoreInput2Layer, self).__init__(name=name)

    def call(self, input1, input2):
        return input1