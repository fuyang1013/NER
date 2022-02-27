"""
globalpointer test unit

Notice:
Since a tensorflow v1 api is used in globalpointer, so we switch to tf.compat.v1 to print
tensor's value

running environment:
    tensorflow==2.4.0
    bert4keras==0.10.8
    torch==1.7.0
"""
import tensorflow.compat.v1 as tf  # since a tf1 api ``
from bert4keras.backend import K
from bert4keras.backend import sequence_masking
from tensorflow.keras.layers import *
import torch
from bert_globalpointer.gp import GlobalPointer


# close tf2 to use session
tf.disable_v2_behavior()


heads = 4
head_size = 6
batch_size = 10
seq_len = 5
emb_dim = 768


# copy code from bert4keras directly
class SinusoidalPositionEmbedding(Layer):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# just copy bert4keras globalpointer code except Dense. Considering the weight is randomly
# initialized, we skip the dense layer to avoid the influence
def tf_gp(inputs, mask):
    inputs = tf.split(inputs, heads, axis=-1)
    inputs = K.stack(inputs, axis=-2)
    qw, kw = inputs[..., :head_size], inputs[..., head_size:]

    # RoPE编码
    pos = SinusoidalPositionEmbedding(head_size, 'zero')(inputs)
    cos_pos = K.repeat_elements(pos[..., None, 1::2], 2, -1)
    sin_pos = K.repeat_elements(pos[..., None, ::2], 2, -1)

    qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
    qw2 = K.reshape(qw2, K.shape(qw))
    qw = qw * cos_pos + qw2 * sin_pos

    kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
    kw2 = K.reshape(kw2, K.shape(kw))
    kw = kw * cos_pos + kw2 * sin_pos

    # 计算内积
    logits = tf.einsum('bmhd,bnhd->bhmn', qw, kw)

    # 排除padding
    logits = sequence_masking(logits, mask, '-inf', 2)
    logits = sequence_masking(logits, mask, '-inf', 3)
    # 排除下三角
    mask = tf.linalg.band_part(K.ones_like(logits), 0, -1)
    logits = logits - (1 - mask) * K.infinity()
    # scale返回
    return logits / head_size**0.5


""" generate the fake input """
inputs = torch.randn([batch_size, seq_len, heads * head_size * 2]).numpy()
mask = torch.ones([batch_size, seq_len]).numpy()


""" calculaton in original tf code """
tfinputs = tf.compat.v1.convert_to_tensor(inputs)
tfmask = tf.compat.v1.convert_to_tensor(mask)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    tfresult = sess.run(tf_gp(tfinputs, tfmask))

print('=====================\n', torch.FloatTensor(tfresult)[0])


""" calculation in pytorch code """
ptinputs = torch.FloatTensor(inputs)
ptmask = torch.IntTensor(mask)
gp = GlobalPointer(emb_dim, seq_len, heads, head_size)
output = gp(ptinputs, ptmask, skip_linear=True)
print('==============\n', output[0])


""" compare the result """
print(
    output - torch.FloatTensor(tfresult), 
    torch.abs(output - torch.FloatTensor(tfresult)).sum()
)