#%%
# [IMPORTANT Libraries Import]
import os
import logging
import numpy as np
import pandas as pd
import cv2
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

from tensorflow import keras

# Set the TensorFlow log level to suppress info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set the logging level to suppress warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# [IMPORTANT Function Declaration]
def formulate_dataset(parent_folder_, no_invert=False):
    X_own_valid = []
    y_own_ = []
    # Loop through the image files in the folder
    for folder_name in os.listdir(parent_folder_):
        folder_path = os.path.join(parent_folder_, folder_name)
        # Check if the file is an image
        if os.path.isdir(folder_path):
            image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".jpg")]
            for i, filename in enumerate(image_files):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(folder_path, filename)
                    first_char = filename[0]
                    y_own_.append(str(first_char))

                    # Open and resize the image
                    image = cv2.imread(image_path)
                    # resized_image = image.resize((64, 64))  # Replace with desired image size

                    # Convert the image to grayscale and flatten it
                    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if not no_invert:
                        grayscale_image = (255 - grayscale_image) # type: ignore
                    _, img_binary_lp = cv2.threshold(grayscale_image, 190, 255, cv2.THRESH_BINARY)

                    # flattened_image = np.array(img_binary_lp).flatten()

                    # Add the flattened image data to the list
                    X_own_valid.append(grayscale_image)

    return np.asarray(X_own_valid).astype(np.float64)/255.0, np.asarray(y_own_)

def fit_pca(X_train, n_components):
    num_samples_train, height, width = X_train.shape
    X_train2d = X_train.reshape(num_samples_train, height * width)

    print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train2d.shape[0]))
    t0 = time()
    # Scale the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train2d)
    
    # Fit PCA on the scaled training data
    pca = RandomizedPCA(n_components=n_components, whiten=True)
    pca.fit(X_train_scaled)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, height, width))
    
    return pca, scaler, eigenfaces

def transform_data(pca, scaler, data):
    num_samples_train, height, width = data.shape
    data_2d = data.reshape(num_samples_train, height * width)

    print("Projecting the input data on the eigenfaces orthonormal basis")
    # Scale and transform training data
    data_scaled = scaler.transform(data_2d)
    data_pca = pca.transform(data_scaled)
    
    return data_pca

class RBF_Model(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, num_prototypes, input_shape, learning_rate, beta):
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.beta = beta
        self.model = None

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        checkpoint = keras.callbacks.ModelCheckpoint(f"./saved_models_v3/{self.input_shape[0]}/rbf_{self.input_shape[0]}_{self.num_prototypes}_model", monitor="val_loss", save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.model = self.build_model(x_train)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                     validation_data=validation_data, callbacks=[checkpoint, early_stopping], verbose=1)
        return history

    def build_model(self, X):
        model = keras.Sequential()

        print("Initializing K-Means clustering to get RBF centers")
        kmeans = KMeans(n_clusters=self.num_prototypes)
        kmeans.fit(X)
        rbf_centers = kmeans.cluster_centers_
        print("RBF centers Initialized")

        model.add(keras.layers.Dense(self.num_prototypes, input_shape=self.input_shape, activation=self.rbf_activation,
                                      kernel_initializer=keras.initializers.Constant(value=rbf_centers),
                                      trainable=True, name='hidden'))

        model.add(keras.layers.Dropout(0.5))  # Add a Dropout layer with a dropout rate of 0.5

        # Apply BatchNormalization layer
        model.add(keras.layers.BatchNormalization())  # Add a BatchNormalization layer

        model.add(keras.layers.Dense(self.num_classes, activation='softmax', name='out'))

        return model

    def rbf_activation(self, x):
        return tf.math.exp(-self.beta * tf.square(x))

    def predict(self, X):
        check_is_fitted(self, "model")
        X = check_array(X)
        return self.model.predict(X).argmax(axis=1)

    def evaluate(self, X, y):
        check_is_fitted(self, "model")
        X = check_array(X)
        y_pred = self.predict(X)
        y_true = y.argmax(axis=1)
        print(classification_report(y_true, y_pred))

    @classmethod
    def load_model(cls, filepath, num_classes, num_prototypes, input_shape, learning_rate, beta):
        model = keras.models.load_model(filepath, custom_objects={"rbf_activation": cls.rbf_activation})
        custom_model = cls(num_classes, num_prototypes, input_shape, learning_rate, beta)
        custom_model.model = model
        return custom_model
    
def k_fold_cross_validation(model_, X, y, batch_size, epochs, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    fold_scores = []
    trained_models = []  # List to store trained models

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Performing Fold: {i+1}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = model_
        history = model.fit(X_train, y_train, batch_size, epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
        fold_scores.append(history.history['val_accuracy'][-1])
        model_history = history.history
        trained_models.append(model)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    return trained_models, model_history, fold_scores
#%%
# [IMPORTANT Data Import And Formulation]
X_parent_folder = "./Datasets/dataset_v3.0/train_test/"
predict_parent_folder = "./Datasets/dataset_v3.0/prediction/"

X, y = formulate_dataset(X_parent_folder, no_invert=True)
X_predict, y_predict = formulate_dataset(predict_parent_folder, no_invert=True)
#%%
# [IMPORTANT Label Encoder]
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(y)
y_predict_encoded = label_encoder.fit_transform(y_predict)

Y_char_classes = np.unique(y)
Y_pred_char_classes = np.unique(y_predict)

nComps = [15, 30, 55, 70, 95]
nProtos = [700, 800, 900, 1000, 1100]

# Create an empty dataframe to store the training logs
training_log = pd.DataFrame(columns=['Model', 'Accuracy', 'Loss', 'Val_Accuracy', 'Val_Loss', 'Training_Time'])

# [IMPORTANT Models training and Export]
for i, nComp_ in enumerate(nComps):
    pca_, scaler_, eigenchars_ = fit_pca(X, nComp_)
    X_pca_ = transform_data(pca_, scaler_, X)
    X_, y_ = check_X_y(X_pca_, Y_encoded)
    y_ = tf.keras.utils.to_categorical(y_, num_classes=36)  # Convert labels to one-hot encoding
    
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42)
    
    for j, nProto_ in enumerate(nProtos):
        model = RBF_Model(num_classes=36, num_prototypes=nProto_, input_shape=(X_pca_.shape[1],), learning_rate=5e-3, beta=1)
        
        start_time = time()
        
        history = model.fit(X_train, y_train, batch_size=32, epochs=1000, validation_data=(X_test, y_test))
        
        end_time = time()
        training_time = end_time - start_time
        
        hist_directory = f"./histories/{nComp_}_history/"
        if not os.path.exists(hist_directory):
            os.makedirs(hist_directory)

        pd.DataFrame(history.history).plot()
        plt.savefig(f'./{hist_directory}/{nProto_}_history.png')
        plt.close()
        
        # Append the training log to the dataframe
        training_log = training_log.append({
            'Model': f'{nComp_}_{nProto_}',
            'Accuracy': history.history['accuracy'][-1],
            'Loss': history.history['loss'][-1],
            'Val_Accuracy': history.history['val_accuracy'][-1],
            'Val_Loss': history.history['val_loss'][-1],
            'Training_Time': training_time
        }, ignore_index=True)

# Export the training log dataframe to CSV
training_log.to_csv('training_log.csv', index=False)
#%%
# [IMPORTANT K-Folds]
# https://isbooktoday.com/freedom/B07TWDNMHJ
parent_directory = "./saved_models_v2/"
# Get the list of child directories
child_directories = next(os.walk(parent_directory))[1]

folds_df = pd.DataFrame()

for i, nComp_ in enumerate(nComps):
    pca_, scaler_, eigenchars_ = fit_pca(X, nComp_)
    X_pca_ = transform_data(pca_, scaler_, X)

    X_, y_ = check_X_y(X_pca_, Y_encoded)
    y_ = tf.keras.utils.to_categorical(y_, num_classes=36)  # Convert labels to one-hot encoding
    child_directory = str(nComp_)  # Assuming the child directory name matches nComp_
    directory_path = os.path.join(parent_directory, child_directory)
    model_files = [file for file in os.listdir(directory_path) if file.startswith("rbf_")]
    print("----------------------------------------------------------------------------------------------------")
    # Iterate over the model files in the directory
    for model_file in model_files:
        model_file_path = os.path.join(directory_path, model_file)

        print(f"Validating Model: {model_file_path}")
        model = keras.models.load_model(model_file_path)
        trained_models, model_history, fold_scores = k_fold_cross_validation(model, X_, y_, batch_size=32, epochs=1000, k=5)

        file_name = os.path.basename(model_file_path)  # Get the file name from the path
        file_name_without_extension = os.path.splitext(file_name)[0]  # Remove the file extension
        values = file_name_without_extension.split("_")  # Split the string at each underscore

        comp = values[1]  # "15"
        proto = values[2]  # "800"

         # Create a dictionary with the model information
        model_info = {
            'component_number': comp,
            'prototype_number': proto
        }

        # Add the fold scores to the dictionary
        for j, score in enumerate(fold_scores):
            model_info[f"Fold {j+1}"] = score

        model_info['mean_score'] = np.mean(fold_scores)
        model_info['std_score'] = np.std(fold_scores)

        # Append the model information to the dataframe
        folds_df = folds_df.append(model_info, ignore_index=True)

output_csv_path = "./folds_data.csv"  # Specify the path and filename for the output CSV file
folds_df.to_csv(output_csv_path, index=False)
#%%
model = keras.models.load_model('./saved_models_v2/30/rbf_30_700_model')
model.summary()

pca_pred_, scaler_, eigenchars_ = fit_pca(X, 30)
X_pca_pred = transform_data(pca_pred_, scaler_, X_predict)

X_p, y_p = check_X_y(X_pca_pred, y_predict_encoded)
y_p = tf.keras.utils.to_categorical(y_p, num_classes=36)  # Convert labels to one-hot encoding

model.evaluate(X_p, y_p)
 
for i in range(10):
    image_idx =  np.random.randint(0, X_p.shape[0])
    X_predict_ = np.reshape(X_p[image_idx], (1, 30))
    
    predictions = model.predict(X_predict_)
    predicted_classes = np.argmax(predictions, axis=1)

    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    print(f"Predicted Label: {predicted_labels}, True Label: {y_predict[image_idx]}")
