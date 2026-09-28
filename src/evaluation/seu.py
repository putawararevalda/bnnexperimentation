import numpy as np
import struct


def float32_to_binary(f):
    [bits] = struct.unpack("!I", struct.pack("!f", f))
    return f"{bits:032b}"


def binary_to_float32(binary_str):
    bits = int(binary_str, 2)
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def bitflip_float32(x, bit_i=None):
    """Flip bit `bit_i` of FP32 scalar or array `x`. Defaults to random bit."""
    if bit_i is None:
        bit_i = np.random.randint(0, 32)

    if hasattr(x, "__iter__"):
        x_ = np.zeros_like(x, dtype=np.float32)
        for i, item in enumerate(x):
            bits = list(float32_to_binary(item))
            bits[bit_i] = "0" if bits[bit_i] == "1" else "1"
            x_[i] = binary_to_float32("".join(bits))
    else:
        bits = list(float32_to_binary(x))
        bits[bit_i] = "0" if bits[bit_i] == "1" else "1"
        x_ = binary_to_float32("".join(bits))

    return x_


def bitflip_float32_with_original(x, bit_i=None):
    """Same as bitflip_float32 but also returns the original bit value."""
    if bit_i is None:
        bit_i = np.random.randint(0, 32)

    if hasattr(x, "__iter__"):
        x_ = np.zeros_like(x, dtype=np.float32)
        original_bit = None
        for i, item in enumerate(x):
            bits = list(float32_to_binary(item))
            original_bit = bits[bit_i]
            bits[bit_i] = "0" if bits[bit_i] == "1" else "1"
            x_[i] = binary_to_float32("".join(bits))
    else:
        bits = list(float32_to_binary(x))
        original_bit = bits[bit_i]
        bits[bit_i] = "0" if bits[bit_i] == "1" else "1"
        x_ = binary_to_float32("".join(bits))

    return x_, original_bit
