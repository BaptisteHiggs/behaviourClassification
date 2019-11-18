import tensorflow as tf
import numpy as np
import random
import json
import sys
import os


class Config():
    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print(f'{a:30} {getattr(self, a)}')
        print()

class SequentialDataset:
    def __init__(self, config):
        self.config = config
        self.classes = []
        self.sequences = []

    def add_class(self, class_name):
        for c in self.classes:
            if class_name == c['name']:
                return
        max_or_zero = lambda l: max(l) if l != [] else -1
        last_id = max_or_zero([c['id'] for c in self.classes])
        self.classes.append({
            'id': last_id + 1,
            'name': class_name
        })

    def add_sequence(self, start_id, end_id, class_id):
        self.sequences.append({
            'start_id': start_id,
            'end_id': end_id,
            'class_id': class_id
        })

    def shuffle_sequences(self):
        random.shuffle(self.sequences)

    def train_val_split(self, val_ratio):
        splitting_index = int(len(self.sequences) * (1 - val_ratio))
        self.train_sequences = self.sequences[:splitting_index]
        self.valid_sequences = self.sequences[splitting_index:]

    def dataset_generator(self, raw_data, mode='all'):
        if 'train_sequences' in list(self.__dict__.keys()):
            assert mode in ['all', 'train', 'valid']
        else:
            assert mode == 'all'

        def sequence_generator(raw_data):
            if mode == 'all':
                sequence = self.sequences
            elif mode == 'train':
                sequence = self.train_sequences
            elif mode == 'valid':
                sequence = self.valid_sequences
            else:
                raise ValueError("mode must be 'all', 'train', or 'valid'")

            for seq in sequence:
                start_idx = seq['start_id']
                end_idx   = seq['end_id']
                sequence  = [d['temps'] for d in raw_data[start_idx:end_idx]]
                sequence  = np.expand_dims(np.array(sequence), axis=1)
                o_h_class = tf.one_hot(seq['class_id'], len(self.classes))
                yield sequence.tolist(), o_h_class.numpy().tolist()

        input_shape = (
            self.config.SEQUENCE_LENGTH,
            self.config.INPUT_FRAME_CHANNELS,
            self.config.INPUT_FRAME_HEIGHT,
            self.config.INPUT_FRAME_WIDTH
        )
        output_shape = (self.config.CLASS_COUNT)

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: sequence_generator(raw_data),
            output_types=(tf.float32, tf.float32),
            output_shapes=(input_shape, output_shape)
        )

        return dataset

class ConvLSTM:
    def __init__(self, model_dir, config):
        self.model_dir = model_dir
        self.config = config
        self.model = self.build()

    def build(self):
        inputs = tf.keras.layers.Input(
            name='inputs',
            shape=(
                config.SEQUENCE_LENGTH,
                config.INPUT_FRAME_CHANNELS,
                config.INPUT_FRAME_HEIGHT,
                config.INPUT_FRAME_WIDTH
            )
        )

        convlstm_params = {
            'kernel_size': (3, 3),
            'data_format': 'channels_first',
            'return_sequences': True
        }

        x = inputs
        for i, filters in enumerate(config.CONVLSTM_FILTERS[:-1]):
            x = tf.keras.layers.ConvLSTM2D(
                name=f'convlstm_{i}',
                filters=filters,
                recurrent_activation='hard_sigmoid',
                activation='tanh',
                **convlstm_params
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'batchnorm_{i}')(x)

        x = tf.keras.layers.ConvLSTM2D(
            name=f'convlstm_{len(config.CONVLSTM_FILTERS)-1}',
            filters=config.CONVLSTM_FILTERS[-1],
            **convlstm_params
        )(x)

        x = tf.keras.layers.Flatten(name='flatten')(x)

        for i, neurons in enumerate(config.DENSE_NEURONS):
            x = tf.keras.layers.Dense(
                name=f'dense_{i}',
                units=neurons,
                activation='relu'
            )(x)

        outputs = tf.keras.layers.Dense(
            name='outputs',
            units=config.CLASS_COUNT
        )(x)

        return tf.keras.Model(inputs, outputs, name='ConvLSTM')

    def summary(self, *args, **kwargs):
        self.model.summary(*args, **kwargs)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


###   ^^^ LIBRARY CODE ABOVE (DON'T CHANGE) ^^^   ###
#####################################################
### vvv USER CODE BELOW (FEEL FREE TO CHANGE) vvv ###

from itertools import combinations

class BathroomConfig(Config):

    # Data Format
    INPUT_FRAME_CHANNELS = 1  ## (int)
        # Because the inputs are from a thermal camera, there is only one
        # channel. If the inputs were from a normal camera (R, G, B), there
        # would be three channels.
    INPUT_FRAME_HEIGHT   = 8  ## (int)
    INPUT_FRAME_WIDTH    = 8  ## (int)
    CLASS_COUNT          = 31  ## (int)
        # The number of "states".

    # Sequence Params
    MAX_TIME_INTERVAL    = 1.5  ## (float) > 0.
        # Maximum "time" between two datapoints to be considered sequential.
    SEQUENCE_LENGTH      = 10    ## (int) > 1
    CLASS_SEQUENCE_INDEX = 2    ## (int) < SEQUENCE_LEN
        # The class at the index within a sequence to be attributed to said
        # sequence.

    # ConvLSTM Params
    CONVLSTM_FILTERS = [32, 16]   ## (list)(int)
    DENSE_NEURONS    = [256, 32]  ## (list)(int)

    # Training Hyperparams
    EPOCHS     = 20    ## (int) >= 1
    BATCH_SIZE = 32    ## (int) >= 1
    VAL_RATIO  = 0.15  ## (float) > 0.; < 1.
        # Percentage of dataset that is set aside for validation.

class BathroomDataset(SequentialDataset):

    def load_sequences_from_json(self, raw_data):

        # Load Classes
        for data in raw_data:
            self.add_class(
                class_name=data['state']
            )

        # Find Indices of Breaks in Data Where Time Interval is Large
        seq_breaks = []
        prev = lambda: seq_breaks[-1][1]
        for i in range(1, len(raw_data)):
            interval_time = raw_data[i]['time'] - raw_data[i-1]['time']
            if interval_time > self.config.MAX_TIME_INTERVAL:
                if seq_breaks == []:
                    seq_breaks.append((0, i))
                else:
                    seq_breaks.append((prev(), i))
        seq_breaks.append((prev(), len(raw_data)))

        # Get Sequence Info for Data Generator
        for i, j in seq_breaks:
            if j - i >= self.config.SEQUENCE_LENGTH:
                for k in range(j - i - self.config.SEQUENCE_LENGTH + 1):

                    # Class ID Based on Sequence Class Index
                    for c in self.classes:
                        c_str = raw_data[i + self.config.CLASS_SEQUENCE_INDEX]
                        if c_str['state'] == c['name']:
                            class_id = c['id']
                            break

                    # Store Sequence
                    self.add_sequence(
                        start_id=i + k,
                        end_id=i + k + self.config.SEQUENCE_LENGTH,
                        class_id=class_id
                    )

        self.shuffle_sequences()
        self.train_val_split(val_ratio=self.config.VAL_RATIO)


def runBatch(raw_data):
    # Build and Compile ConvLSTM Model
    model = ConvLSTM(MODEL_DIR, config)
    model.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.MeanSquaredError(),
        metrics=['acc']
    )
    print(f'\n{model.summary()}')

    # Load Dataset
    dataset = BathroomDataset(config)
    dataset.load_sequences_from_json(raw_data)

    # Preprocess Datasets
    train_ds   = dataset.dataset_generator(raw_data, 'train')
    valid_ds   = dataset.dataset_generator(raw_data, 'valid')
    preprocess = lambda inp, out: (inp/25, out)
    prepare_ds = lambda ds: ds.map(preprocess).repeat().batch(config.BATCH_SIZE)

    # Train
    dp = len(dataset.sequences)
    train_steps = dp * (1 - config.VAL_RATIO) // config.BATCH_SIZE + 1
    valid_steps = dp * config.VAL_RATIO // config.BATCH_SIZE + 1
    print(f'Train for {dp} datapoints in batches of {config.BATCH_SIZE}')
    history = model.fit(
        prepare_ds(train_ds),
        steps_per_epoch=train_steps,
        epochs=config.EPOCHS,
        verbose=2,
        validation_data=prepare_ds(valid_ds),
        validation_steps=valid_steps
    )
    
    # Finding the highest accuracy developed
    highestAccuracy = max(history.history['val_acc'])
    return highestAccuracy*100

if __name__ == '__main__':
    # Set file locations
    JSON_PATH = 'behaviourData.json'
    MODEL_DIR = 'models'

    # Set config
    config = BathroomConfig()
    print(config.display())

    # Read .json File to extract behaviours
    with open(JSON_PATH, 'r') as json_file:
        # Load file
        raw_data = json.load(json_file)

    highestAccuracy = runBatch(raw_data)



### TODO:
# [x] Write a class to parse json data into sequences and one hot classes
# [x] Write a class to build a ConvLSTM model
# [x] Write a generator to load data from json for training
# [x] Update generator to split data into training and validation dataset
# [ ] Update training method to incrementally save models
# [ ] Write a pipeline to make inferences on a trained model

### NOTE:
# 01. There are 31 large gaps in your data, ranging from a time interval of
#     2.856 to 11367.564 time units. Within the method that parses the data,
#     I've added an argument that allows you to control the maximum time
#     between data points for the data to be considered sequential.