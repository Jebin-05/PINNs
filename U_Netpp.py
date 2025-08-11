import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    """A basic convolutional block: Conv -> Conv -> Dropout"""
    x = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    return x

def build_unetpp(input_shape=(128, 128, 1), num_classes=1):
    """
    Lightweight UNet++ implementation for CPU-friendly training.
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 16)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 128)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 256)

    # Decoder with skip connections (simplified UNet++)
    u4 = UpSampling2D((2, 2))(c5)
    u4 = Concatenate()([u4, c4])
    c6 = conv_block(u4, 128)

    u3 = UpSampling2D((2, 2))(c6)
    u3 = Concatenate()([u3, c3])
    c7 = conv_block(u3, 64)

    u2 = UpSampling2D((2, 2))(c7)
    u2 = Concatenate()([u2, c2])
    c8 = conv_block(u2, 32)

    u1 = UpSampling2D((2, 2))(c8)
    u1 = Concatenate()([u1, c1])
    c9 = conv_block(u1, 16)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    return model
