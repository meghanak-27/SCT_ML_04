# SCT_ML_04
Gesture Recognition using CNN
This project focuses on classifying hand gesture images using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras. The dataset used is Leap Gestures Recognition, containing different gesture categories (e.g., palm, thumb) that are recognized by the model.

Project Structure-

>Data Loading: The dataset is organized into folders representing different gesture classes. Images are loaded, resized to 64x64 pixels, and normalized for model training.

>Data Preprocessing: The image data is augmented to enhance the diversity of the training dataset. Labels are encoded as integers to represent each gesture class.

>Model Architecture: A CNN with multiple layers is built, including convolutional layers, max-pooling layers, dropout for regularization, and fully connected layers. The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

>Model Evaluation: The model's performance is evaluated on the test data using accuracy, and training and validation metrics are visualized using plots.
