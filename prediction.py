import joblib

# Load the savedmodel
model = joblib.load('model/iris_rf_model.pkl')


def predictor(features):
    return model.predict(features)
