import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from emnist import extract_training_samples, extract_test_samples
from keras_tuner.tuners import BayesianOptimization
import os



def load_data():
    print("Loading data...")
    train_images, train_labels = extract_training_samples('byclass')
    test_images, test_labels = extract_test_samples('byclass')
    return train_images, train_labels, test_images, test_labels


def preprocess_data(train_images, test_images):
    print("Preparing data...")
    train_images = tf.keras.utils.normalize(train_images, axis=1)
    test_images = tf.keras.utils.normalize(test_images, axis=1)
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    return train_images, test_images


def create_data_generator(train_images, test_images): 
    print("Creating data generator...")

    rotation_range_val = 15
    width_shift_val = 0.10
    height_shift_val = 0.10

    train_datagen = ImageDataGenerator(
        rotation_range=rotation_range_val,
        width_shift_range=width_shift_val,
        height_shift_range=height_shift_val
    )
    train_datagen.fit(train_images)

    val_datagen = ImageDataGenerator()
    val_datagen.fit(test_images)

    return train_datagen, val_datagen


def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(28, 28, 1)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=hp.Int('dense1_units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))
    model.add(layers.Dense(
        units=hp.Int('dense2_units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))
    model.add(layers.Dense(62, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def train_model(build_model, train_datagen, val_datagen, train_images, train_labels, test_images, test_labels):   
    print("Training model...")
    # dir for temp files
    current_directory = os.path.abspath(os.getcwd()).replace("\\", "/")

    # Define the tuner and perform hyperparameter search
    tuner = BayesianOptimization(build_model,
                                 objective='val_accuracy',
                                 max_trials=12,
                                 executions_per_trial=1,
                                 directory=current_directory,
                                 project_name='mnist_tuning',
                                 overwrite=True)

    tuner.search(train_datagen.flow(train_images, train_labels), 
                 validation_data=val_datagen.flow(test_images, test_labels), 
                 epochs=10)

    return tuner


def save_model(model):
    print("Saving model...")
    model.save("./app/model.h5")
    print("Model successfully trained and saved.")
    

def train_mnist_model():
    train_images, train_labels, test_images, test_labels = load_data()
    train_images, test_images = preprocess_data(train_images, test_images)
    train_datagen, val_datagen = create_data_generator(train_images, test_images)

    # normalize labels
    train_labels = train_labels
    test_labels = test_labels
    

    # Retrieve the best model and evaluate it on the test data
    tuner = train_model(build_model, train_datagen, val_datagen, train_images, train_labels, test_images, test_labels)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

    print("Best Hyperparameters:")
    print(best_hyperparameters.get_config())
    print("Best Model Evaluation:")
    print(best_model.evaluate(test_images, test_labels))

    # Save the best model
    save_model(best_model)

# Run the training process
train_mnist_model()
