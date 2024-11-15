from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf  # type: ignore

# wav files (trainingdataset)
files = ["testdata/test.wav", "testdata/test.wav", "testdata/test.wav"]

# mental labels
labels = ["FLAT", "HAPPY", "SAD"]

# read wav audio file and return binary data
def read_binary(file):
    return tf.io.read_file(file)

# decode wav binary data
def decode_wav(binary):
    audio_data, _ = tf.audio.decode_wav(binary)
    return audio_data

# compute rms
# (sound pressure level)
def compute_rms(data):
    return tf.sqrt(tf.reduce_mean(tf.square(data))).numpy()

# compute rate
# (calculates the zero-crossing rate of the data. This indicates how often the waveform changes to positive or negative)
def compute_rate(data):
    return tf.reduce_mean(tf.cast(data[1:] * data[:-1] < 0, tf.float32)).numpy()

# get feature. return rms and rate
# (extracts the RMS value and zero crossing rate of the specified file and returns them as features)
def extract_feature(file):
    binary = read_binary(file)
    data = decode_wav(binary)
    return [compute_rms(data), compute_rate(data)]

# get features
features = [extract_feature(f) for f in files]

standardscaler = StandardScaler()

features = standardscaler.fit_transform(features)

encoder = LabelEncoder()

labels_encoded = encoder.fit_transform(labels)

# all flat: single checks
if len(set(labels_encoded)) == 1:
    def predict(input_data):
        return labels[0]  # all "flat"
else:
    # create tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels_encoded)).batch(10)

    # build model
    act_relu = "relu"
    act_softmax = "softmax"
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(features[0]),)),
        tf.keras.layers.Dense(128, activation=act_relu),
        tf.keras.layers.Dense(64, activation=act_relu),
        tf.keras.layers.Dense(32, activation=act_relu),
        tf.keras.layers.Dense(len(set(labels)), activation=act_softmax)
    ])


    opt = "adam"
    
    loss = "sparse_categorical_crossentropy"
    
    metric="accuracy"

    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    
    model.fit(dataset, epochs=10)

    # prediction
    def predict(input):
        data = standardscaler.transform([input])  

        prediction = model.predict(data)
        
        predicted_label = encoder.inverse_transform([tf.argmax(prediction[0]).numpy()])[0]
        
        return predicted_label



if __name__ == "__main__":
    print(f"label: {predict(extract_feature("testdata/test.wav"))}")