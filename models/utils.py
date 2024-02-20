import tensorflow as tf
import numpy as np

@tf.function
def logit_to_binary(x):
    return (tf.math.sign(x) + 1.0) / 2.0

@tf.function
def tensor_to_binary_u32(x):
    # convert to bits
    bits = [None for i in range(32)]
    for i in range(32):
        bits[i] = tf.bitwise.bitwise_and(x, 1)
        x = tf.bitwise.right_shift(x, 1)
    res = tf.stack(bits, axis=-1)
    return tf.cast(res, tf.float32)

@tf.function
def tensor_to_binary_v2(x):
    LOG_BASE2 = tf.math.log(tf.constant([2.0], tf.float32))
    TO_MANTISSA = tf.constant([1<<23], tf.float32)

    # sign
    sign = tf.cast(x < 0.0, tf.float32)
    x = tf.math.abs(x)
    
    # exponent
    log_x = tf.math.floor(tf.math.log(x) / LOG_BASE2)
    exponent = tf.cast(log_x + 127.0, tf.uint8)
    
    # mantissa
    mantissa = x / tf.math.exp(log_x*LOG_BASE2) - tf.math.sign(x)
    mantissa = tf.math.floor(mantissa * TO_MANTISSA)
    mantissa = tf.cast(mantissa, tf.int32)

    # convert to bits
    bits = [None for i in range(32)]
    for i in range(23):
        bits[i] = tf.bitwise.bitwise_and(mantissa, 1)
        mantissa = tf.bitwise.right_shift(mantissa, 1)
    for i in range(23, 31):
        bits[i] = tf.bitwise.bitwise_and(exponent, 1)
        exponent = tf.bitwise.right_shift(exponent, 1)
    bits[31] = sign

    for i in range(32):
        bits[i] = tf.cast(bits[i], tf.float32)
    res = tf.stack(bits, axis=-1)
    return res

@tf.function
def binary_to_tensor_u32(x: tf.Tensor):
    x = tf.reshape(x, (-1, 32))
    x = tf.cast(x, tf.uint32)
    out = x[:, 0]
    for i in range(1, 32):
        out += tf.bitwise.left_shift(x[:, i], i)
    return out

@tf.function
def binary_to_tensor_v2(x: tf.Tensor):
    LOG_BASE2 = tf.math.log(tf.constant([2.0], tf.float32))
    EXPONENTS = tf.constant([ float(1 << i) for i in range(8)], tf.float32)
    FROM_MANTISSA = tf.constant([ 0.5**(23-i) for i in range(23)], tf.float32)

    x = tf.reshape(x, (-1, 32))
    sign = -x[:, 31] * 2 + 1

    exponent = tf.math.reduce_sum(x[:, 23:31] * EXPONENTS, axis=-1)
    mantissa = tf.math.reduce_sum(x[:, :23] * FROM_MANTISSA, axis=-1)
    mantissa += tf.cast(exponent > 0.0, tf.float32)
    return sign * tf.math.exp((exponent - 127.0) * LOG_BASE2) * mantissa

@tf.function
def tensor_to_binary(x: tf.Tensor):
    x = tf.bitcast(x, tf.uint32)
    mask = tf.ones_like(x)
    bit0 = tf.cast(tf.reshape(tf.bitwise.bitwise_and(x, mask), (1, -1)),
                   tf.float32)
    bits = [bit0]

    for _ in range(31):
        x = tf.bitwise.right_shift(x, 1)
        bitn = tf.cast(tf.reshape(tf.bitwise.bitwise_and(x, mask), (1, -1)),
                       tf.float32)
        bits.append(bitn)

    return tf.concat(bits, axis=0)

@tf.function
def binary_to_tensor(x: tf.Tensor):
    x = tf.cast(x, tf.uint32)
    x = tf.reshape(x, (32, -1))

    shape = tf.shape(x)
    out = tf.zeros((shape[1],), dtype=tf.uint32)
    for i in range(32):
        bitn = tf.bitwise.left_shift(x[i, :], i)
        out = tf.bitwise.bitwise_xor(out, bitn)

    return tf.bitcast(out, tf.float32)


@tf.function
def replace_nan(input, new_value = 0.0):
    new_value = float(new_value)
    indices = tf.where(tf.math.is_nan(input))
    res = tf.tensor_scatter_nd_update(
        input,
        indices,
        tf.fill((tf.shape(indices)[0], ), new_value)
    )
    return res

INF = tf.constant(np.array([np.inf]), dtype=tf.float32)

@tf.function
def replace_nan_to_inf(x):
    EXPONENT_MASK = tf.constant([0x7F800000], dtype=tf.uint32)
    INF_MASK = tf.constant([0xFF800000], dtype=tf.uint32)
    IDENTITY_MASK = tf.constant([0xFFFFFFFF], dtype=tf.uint32)
    x = tf.bitcast(x, tf.uint32)
    mask = tf.where(
        tf.equal(tf.bitwise.bitwise_and(x, EXPONENT_MASK), EXPONENT_MASK),
        INF_MASK, IDENTITY_MASK
    )
    return tf.bitcast(tf.bitwise.bitwise_and(x, mask), tf.float32)

@tf.function
def get_ber(b, b_hat):
    bit_errors = tf.math.count_nonzero(b != b_hat, dtype=tf.float32)
    total_bits = tf.cast(tf.reduce_prod(b.shape), dtype=tf.float32)
    ber = bit_errors / total_bits

    return ber

def test():
    # f32
    x = tf.random.normal(shape=(64,))
    b = tensor_to_binary_v2(x)
    y = binary_to_tensor_v2(b)
    tf.debugging.assert_near(x, y)

    # 
    x = tf.random.uniform(shape=(64,), maxval=tf.int32.max, dtype=tf.int32)
    x = tf.cast(x, tf.uint32)
    b = tensor_to_binary_u32(x)
    y = binary_to_tensor_u32(b)
    tf.debugging.assert_equal(x, y)

if __name__ == '__main__':
    test()