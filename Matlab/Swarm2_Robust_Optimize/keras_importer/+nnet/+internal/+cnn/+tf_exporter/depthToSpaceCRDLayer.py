class depthToSpaceCRDLayer(tf.keras.layers.Layer):
    def __init__(self, blocksize, name=None):
        super(depthToSpaceCRDLayer, self).__init__(name=name)
        self.blocksize = blocksize

    def call(self, input):
        nhwc = tf.shape(input)
        batchSize= nhwc[0]
        inputHeight = nhwc[1]
        inputWidth = nhwc[2]
        inputChannel = nhwc[3]
        outputChannel = tf.cast(inputChannel/(self.blocksize*self.blocksize), tf.int32)
        outputHeight = tf.cast(inputHeight*self.blocksize, tf.int32)
        outputWidth = tf.cast(inputWidth*self.blocksize, tf.int32)
        
        x = tf.transpose(input, [0,3,2,1])                                  # transpose to reverse MATLAB
        x = tf.reshape(x, [batchSize, outputChannel, self.blocksize, self.blocksize, inputWidth, inputHeight])
        x = tf.transpose(x, [0,1,4,3,5,2])
        x = tf.reshape(x, [batchSize, outputChannel, outputWidth, outputHeight])
        x = tf.transpose(x, [0,3,2,1])                                      # transpose back from reverse MATLAB
        return x