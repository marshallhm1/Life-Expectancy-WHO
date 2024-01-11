import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Function to load data from a CSV file
def load_data(file_path='life_expectancy.csv'):
    dataset = pd.read_csv(file_path)
    dataset = dataset.drop(['Country'], axis=1)
    labels = dataset.iloc[:, -1]
    features = dataset.iloc[:, 0:-1]
    features = pd.get_dummies(dataset)
    return features, labels

# Function to preprocess the data, including scaling and train/test split
def preprocess_data(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20,
                                                                                random_state=23)

    numerical_features = features.select_dtypes(include=['float64', 'int64'])
    numerical_columns = numerical_features.columns

    ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

    features_trained_scaled = ct.fit_transform(features_train)
    features_test_scaled = ct.transform(features_test)

    return features_trained_scaled, features_test_scaled, labels_train, labels_test, ct

# Function to build the neural network model
def build_model(input_shape):
    model = Sequential()
    input_layer = InputLayer(input_shape=input_shape)
    model.add(input_layer)
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1))

    opt = Adam(learning_rate=0.001)

    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

# Function to train the model and save it to a file
def train_and_save_model(model, features_train_scaled, labels_train, file_path='life_expectancy.keras', epochs=20,
                         batch_size=1):
    history = model.fit(features_train_scaled, labels_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save the trained model
    model.save(file_path)

    return history

if __name__ == "__main__":
    # Load data
    features, labels = load_data()
    features_train_scaled, features_test_scaled, labels_train, labels_test, ct = preprocess_data(features, labels)

    # Build the model
    input_shape = (features.shape[1],)
    model = build_model(input_shape)

    # Train and save the model
    history = train_and_save_model(model, features_train_scaled, labels_train)

    # Plot training metrics
    plt.figure(figsize=(12, 4))

    # Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'])
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')

    # Training Mean Absolute Error (MAE)
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'])
    plt.title('Training MAE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')

    # Scatter Plot of Predictions vs. Actuals
    plt.subplot(1, 3, 3)
    predictions = model.predict(features_test_scaled)
    plt.scatter(labels_test, predictions)
    plt.title('Actual vs Predicted Life Expectancy')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')

    plt.tight_layout()

    # Save the figure
    plt.savefig('training_metrics_and_predictions.png')
    plt.show()

    # Residuals Analysis
    residuals = labels_test.values.flatten() - predictions.flatten()

    # Residuals Histogram
    plt.hist(residuals, bins=30)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # Save the figure
    plt.savefig('residuals_histogram.png')
    plt.show()

    # Q-Q Plot of Residuals
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals')

    # Save the figure
    plt.savefig('qq_plot_residuals.png')
    plt.show()
