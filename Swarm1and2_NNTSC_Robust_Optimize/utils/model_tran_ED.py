import tensorflow as tf
from tensorflow import keras
import numpy as np

## TRANSFORMER (subclass model)
def tr_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    # kernel_initializer=hparams.kernel_initializer
    # dropout=hparams.dropout

    num_layers = 4 # repeat encoder
    # d_model = 128
    d_model = input_shape[1]
    print(f'd_model {d_model}')
    dff = 512
    num_heads = 8
    dropout= 0.1
    embed_type='cn'

    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout,
        embed_type=embed_type,
        out_activation=out_activation,
        output_shape=output_shape,
        name='Trans_' + hparams.output_type
        )

    return transformer

def positional_encoding(length, depth):
    if depth % 2 == 1: depth += 1  # depth must be even
    depth = depth/2 # halve: 1/2 for SIN and COS
    positions = np.arange(length)[:, np.newaxis]     # (length, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth/2)
    angle_rates = 1 / (10000**depths)         # (1, depth/2)
    angle_rads = positions * angle_rates      # (length, depth/2)
    pos_encoding = np.concatenate(
       [np.sin(angle_rads), np.cos(angle_rads)],axis=-1) #(length, depth)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                #  vocab_size,
                 d_model, embed_type='fc'):
        super().__init__()
        self.d_model = d_model
        if embed_type == 'lstm':
            self.embed = tf.keras.layers.LSTM(units=d_model, return_sequences=True)
        elif embed_type == 'fc':
            self.embed = tf.keras.layers.Dense(units=d_model)
        elif embed_type== 'cn':
            self.embed = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, activation="relu")
        else:
            raise Exception("unknown embed_type = {}".format(embed_type))
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        # x = tf.cast(x, tf.float32) #fix type issue for labels?
        x = self.embed(x)
        # print(f'x embed {x}')
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # print(f'x sqrt {x}')
        # print(f'pos encod {self.pos_encoding[tf.newaxis, :length, :self.d_model]}')
        x = x + self.pos_encoding[tf.newaxis, :length, :self.d_model] # :self.d_model
        return x

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        return x
    
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
            ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff,
                 #  vocab_size,
                 dropout_rate=0.1, embed_type='fc'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            # vocab_size=vocab_size,
            d_model=d_model,
            embed_type=embed_type)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x) # Shape `(batch_size, seq_len, d_model)`
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super().__init__() #super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x) #(x=x)
        x = self.cross_attention(x, context) #(x=x, context=context)
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 #  vocab_size,
                 dropout_rate=0.1, embed_type='fc'):
        super().__init__() #super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            # vocab_size=vocab_size,
            d_model=d_model,
            embed_type=embed_type)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        # x = tf.cast(x, tf.float32) #fix type issue for labels?
        x = tf.expand_dims(x, -1) #add "dimension" axis = 1
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x # shape '(batch_size, target_seq_len, d_model)'

# # ENCODER ONLY
# class Transformer(tf.keras.Model):
#     def __init__(self, num_layers, d_model, num_heads, dff, 
#                   dropout_rate, embed_type,
#                   out_activation, output_shape, name):
#         super().__init__(name=name)
#         self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
#                                num_heads=num_heads, dff=dff,
#                                dropout_rate=dropout_rate, embed_type=embed_type)
#         self.gap = tf.keras.layers.GlobalAveragePooling1D()
#         self.final_layer = tf.keras.layers.Dense(output_shape, activation=out_activation)

#     def call(self, inputs):
#         x = self.encoder(inputs)  # (batch_size, data_len, data_dim)
#         x = self.gap(x) # global average pooling
#         output = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
#         return output # Return the final output and the attention weights
    
#     def get_config(self):
#         base_config = super().get_config()
#         return base_config

ENCODER & DECODER
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                #  input_vocab_size, target_vocab_size, 
                  dropout_rate, embed_type,
                  out_activation, output_shape, name):
        super().__init__(name=name)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                            #    vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate, embed_type=embed_type)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                            #    vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate, embed_type=embed_type)
        self.final_layer = tf.keras.layers.Dense(output_shape, activation=out_activation)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs
        # print("\n*** ENCODER ***")
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        # print("\n*** DECODER ***")
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        output = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        # try:
        #     del logits._keras_mask # Drop the keras mask, so it doesn't scale the losses/metrics
        # except AttributeError:
        #     pass
        return output # Return the final output and the attention weights
    
    def get_config(self):
        base_config = super().get_config()
        return base_config