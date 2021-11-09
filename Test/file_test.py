import tensorflow as tf
import numpy as np

with open("Datasets/victorhugo.txt", "r") as f:
    text = f.read()

print(len(text))

print(text[:1000])

import unidecode

text = unidecode.unidecode(text)
text = text.lower()

text = text.replace("2", "")
text = text.replace("1", "")
text = text.replace("8", "")
text = text.replace("5", "")
text = text.replace(">", "")
text = text.replace("<", "")
text = text.replace("!", "")
text = text.replace("?", "")
text = text.replace("-", "")
text = text.replace("$", "")

text = text.strip()

vocab = set(text)
print(len(vocab), vocab)

print(text[:1000])


vocab_size = len(vocab)

vocab_to_int = {l:i for i,l in enumerate(vocab)}
int_to_vocab = {i:l for i,l in enumerate(vocab)}

print("vocab_to_int", vocab_to_int)
print()
print("int_to_vocab", int_to_vocab)

print("\nint for e:", vocab_to_int["e"])
int_for_e = vocab_to_int["e"]
print("letter for %s: %s" % (vocab_to_int["e"], int_to_vocab[int_for_e]))

encoded = [vocab_to_int[l] for l in text]
encoded_sentence = encoded[:100]

print(encoded_sentence)

decoded_sentence = [int_to_vocab[i] for i in encoded_sentence]
print(decoded_sentence)

decoded_sentence = "".join(decoded_sentence)
print(decoded_sentence)

inputs, targets = encoded, encoded[1:]

print("Inputs", inputs[:10])
print("Targets", targets[:10])


def gen_batch(inputs, targets, seq_len, batch_size, noise=0):
    # Size of each chunk
    chuck_size = (len(inputs) - 1) // batch_size
    # Numbef of sequence per chunk
    sequences_per_chunk = chuck_size // seq_len

    for s in range(0, sequences_per_chunk):
        batch_inputs = np.zeros((batch_size, seq_len))
        batch_targets = np.zeros((batch_size, seq_len))
        for b in range(0, batch_size):
            fr = (b * chuck_size) + (s * seq_len)
            to = fr + seq_len
            batch_inputs[b] = inputs[fr:to]
            batch_targets[b] = inputs[fr + 1:to + 1]

            if noise > 0:
                noise_indices = np.random.choice(seq_len, noise)
                batch_inputs[b][noise_indices] = np.random.randint(0, vocab_size)

        yield batch_inputs, batch_targets


for batch_inputs, batch_targets in gen_batch(inputs, targets, 5, 32, noise=0):
    print(batch_inputs[0], batch_targets[0])
    break

for batch_inputs, batch_targets in gen_batch(inputs, targets, 5, 32, noise=3):
    print(batch_inputs[0], batch_targets[0])
    break


class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)


class RnnModel(tf.keras.Model):

    def __init__(self, vocab_size):
        super(RnnModel, self).__init__()
        # Convolutions
        self.one_hot = OneHot(len(vocab))

    def call(self, inputs):
        output = self.one_hot(inputs)
        return output


batch_inputs, batch_targets = next(gen_batch(inputs, targets, 50, 32))

print(batch_inputs.shape)

model = RnnModel(len(vocab))
output = model.predict(batch_inputs)

print(output.shape)

#print(output)

print("Input letter is:", batch_inputs[0][0])
print("One hot representation of the letter", output[0][0])


vocab_size = len(vocab)

### Creat the layers

# Set the input of the model
tf_inputs = tf.keras.Input(shape=(None,), batch_size=64)
# Convert each value of the  input into a one encoding vector
one_hot = OneHot(len(vocab))(tf_inputs)
# Stack LSTM cells
rnn_layer1 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(rnn_layer1)
# Create the outputs of the model
hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)
outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(hidden_layer)

### Setup the model
model = tf.keras.Model(inputs=tf_inputs, outputs=outputs)


# Star by resetting the cells of the RNN
model.reset_states()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # Make a prediction on all the batch
        predictions = model(inputs)
        # Get the error/loss on these predictions
        loss = loss_object(targets, predictions)
    # Compute the gradient which respect to the loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # The metrics are accumulate over time. You don't need to average it yourself.
    train_loss(loss)
    train_accuracy(targets, predictions)


def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions


model.reset_states()

for epoch in range(4000):
    for batch_inputs, batch_targets in gen_batch(inputs, targets, 100, 64, noise=13):
        train_step(batch_inputs, batch_targets)
    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()



# save the model

import json
model.save("model_rnn.h5")

with open("model_rnn_vocab_to_int", "w") as f:
    f.write(json.dumps(vocab_to_int))
with open("model_rnn_int_to_vocab", "w") as f:
    f.write(json.dumps(int_to_vocab))


# Generate some text

import random

model.reset_states()

size_poetries = 300

poetries = np.zeros((64, size_poetries, 1))
sequences = np.zeros((64, 100))
for b in range(64):
    rd = np.random.randint(0, len(inputs) - 100)
    sequences[b] = inputs[rd:rd+100]

for i in range(size_poetries+1):
    if i > 0:
        poetries[:,i-1,:] = sequences
    softmax = predict(sequences)
    # Set the next sequences
    sequences = np.zeros((64, 1))
    for b in range(64):
        argsort = np.argsort(softmax[b][0])
        argsort = argsort[::-1]
        # Select one of the strongest 4 proposals
        sequences[b] = argsort[0]

for b in range(64):
    sentence = "".join([int_to_vocab[i[0]] for i in poetries[b]])
    print(sentence)
    print("\n=====================\n")



# use saved model

import json

with open("model_rnn_vocab_to_int", "r") as f:
    vocab_to_int = json.loads(f.read())
with open("model_rnn_int_to_vocab", "r") as f:
    int_to_vocab = json.loads(f.read())
    int_to_vocab = {int(key):int_to_vocab[key] for key in int_to_vocab}

model.load_weights("model_rnn.h5")