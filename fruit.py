# Import necessary libraries
import tensorflow as tf
from tensorflow import keras

"""
These lines import the required libraries: tensorflow and keras,
which are used for building and training deep learning models.
"""

# Define the paths to your dataset folders
train_data_dir = 'MY_data/train'
test_data_dir = 'MY_data/test'

"""
These lines specify the paths to the folders containing the training and test datasets.
Make sure to replace 'MY_data/train' and 'MY_data/test' with the actual paths to your train
and test data folders.
"""

# Data preprocessing and augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


"""
These lines create instances of ImageDataGenerator, which perform data preprocessing and augmentation.
Here, we only apply rescaling by dividing the pixel values by 255 to normalize them to the range of [0, 1].
"""

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),  # Assuming you want images resized to 150x150
    batch_size=32,
    class_mode='categorical'
)


"""
This code loads and preprocesses the training dataset using the flow_from_directory method of ImageDataGenerator.
It reads the images from the specified directory (train_data_dir), resizes them to the target size of 150x150 pixels,
creates batches of size 32, and uses one-hot encoding for multi-class classification.
"""

# Load and preprocess the test dataset
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

"""
Similarly, this code loads and preprocesses the test dataset using the same configuration as the training dataset.
"""

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # Assuming you have 10 fruit classes
])

"""
This code defines the model architecture using the Sequential class from keras.
It starts with a convolutional layer (Conv2D) with 32 filters of size 3x3, followed by a ReLU activation function.
The input shape is set to (150, 150, 3) to match the size of the input images.
Then, a max pooling layer (MaxPooling2D) with a pool size of 2x2 is applied.
This pattern is repeated with additional convolutional and max pooling layers to extract deeper features.
Next, the feature maps are flattened using Flatten.
Finally, two dense layers (Dense) are added with ReLU and softmax activation functions, respectively.
Adjust the number of units in the last dense layer to match the number of fruit classes in your dataset.
"""

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""
This code compiles the model. The adam optimizer is used with a categorical cross-entropy loss function,
which is suitable for multi-class classification problems.
The accuracy metric is also specified to evaluate the model's performance during training.
"""


# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

"""
This code trains the model using the fit method.
It takes the train_generator as the training data, performs training for 10 epochs,
and uses the test_generator as the validation data to monitor the model's performance during training.
"""

import numpy as np
from tensorflow import keras
model.save('saved_model.h5')
# Load the trained model
model = keras.models.load_model('saved_model.h5')

"""
These lines save the trained model to a file named 'saved_model.h5' using the save method.
Later, the model is loaded from the saved file using the load_model method.
"""

# Load and preprocess the input image(s)
image_paths = ['image1.jpg', 'image2.jpg']  # Replace with your own image paths
images = []
for path in image_paths:
    img = keras.preprocessing.image.load_img(path, target_size=(150, 150))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    images.append(img)

    
"""
This code loads and preprocesses the input image(s) specified in the image_paths list.
Each image is loaded using load_img from keras.preprocessing.image, resized to the target size of 150x150 pixels,
converted to a NumPy array using img_to_array, expanded the dimensions to match the input shape of the model,
and normalized by dividing by 255. The preprocessed image(s) are then stored in the images list.
"""

class_names = ['Apple', 'avocado', 'Banana', 'cherry', 'kiwi', 'mango','orange', 'pineapple', 'strawberries','watermelon']


"""
This line defines the list class_names which contains the names of the fruit classes
in the same order as their corresponding indices. Adjust the class names based on your dataset.
"""


# Make predictions
for i, img in enumerate(images):
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    
    print("Image:", image_paths[i])
    print("Predicted class index:", class_index)
    print("Predicted class:", class_names[class_index])
    print()

    
"""
This code makes predictions using the loaded model for each image in the images list.
It uses model.predict to obtain the predicted probabilities for each class.
The argmax function is used to find the index of the class with the highest probability.
Then, the predicted class index, the corresponding class name retrieved from class_names, and the image path are printed to the console.
"""






