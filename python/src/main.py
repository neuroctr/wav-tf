import tensorflow as tf  # type: ignore
import numpy as np

# wav file path
wav_file_path = "testdata/test.wav"

# read as binary
binary = tf.io.read_file(wav_file_path)

# decode wav file. returns 1: audio data, 2: sample rate
audio_data, sample_rate = tf.audio.decode_wav(binary)

# cast float32 (for type compatibility)
audio = tf.cast(audio_data, tf.float32)

# Compute RMS
rms = tf.sqrt(tf.reduce_mean(tf.square(audio)))
print("RMS - ", rms.numpy())

# cast dB scale
rms_db = 20 * tf.math.log(rms + 1e-6) / tf.math.log(10.0)
print("dB - ", rms_db.numpy())

# print sample rate
print("Sample rate - :", sample_rate.numpy())