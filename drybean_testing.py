import pandas as pd
import joblib

# Load saved model
saved_data = joblib.load("dry_bean_model.joblib")

model = saved_data["model"]
le = saved_data["label_encoder"]
features = saved_data["features"]

# Enter new data
new_data = pd.DataFrame([{
    "Area": 28395,
    "Perimeter": 610.291,
    "MajorAxisLength": 208.178117,
    "MinorAxisLength": 173.888747,
    "AspectRation": 1.197191,
    "Eccentricity": 0.549812,
    "ConvexArea": 28715,
    "EquivDiameter": 190.141097,
    "Extent": 0.763923,
    "Solidity": 0.988856,
    "roundness": 0.958027,
    "Compactness": 0.913358,
    "ShapeFactor1": 0.007332,
    "ShapeFactor2": 0.003147,
    "ShapeFactor3": 0.834222,
    "ShapeFactor4": 0.998724
}])

# Arrange features
new_data = new_data[features]

# Predict
prediction = model.predict(new_data)

# Convert prediction into original class
predicted_class = le.inverse_transform(prediction)

print("New Data:")
print(new_data)

print("Predicted Dry Bean Class:", predicted_class[0])

print("Model is working successfully!")
