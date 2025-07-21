import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv('data/Iris.csv')

df = df.drop(columns=['Id'])
df['Species'] = df['Species'].str.replace('Iris-', '', regex=False)


X = df [['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y = df ['Species']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=15, random_state=42)
model.fit(X_train, Y_train)


joblib.dump(model, 'model/iris_rf_model.pkl')

print("Model trained and saved as 'iris_rf_model.pkl'")