import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
load_best_model = pickle.load(open("../data/best_model_CKD.sav", 'rb'))

# Define test data to match training feature names
test_data_dict = {
    "age": [50],  # Age
    "bp": [76.45],  # Blood Pressure
    "al": [0],  # Albumin
    "su": [0],  # Sugar
    "bgr": [148.11],  # Blood Glucose Random
    "bu": [25],  # Blood Urea
    "sc": [0.6],  # Serum Creatinine
    "sod": [137.52],  # Sodium
    "pot": [4.62],  # Potassium
    "hrmo": [11.8],  # Hemoglobin
    "pcv": [36],  # Packed Cell Volume
    "wc": [12400],  # White Blood Cell Count
    "rc": [4.70],  # Red Blood Cell Count
    "sg_b": [0],  # Encoded specific gravity
    "sg_c": [0],  # One-hot encoded SG levels
    "sg_d": [0],
    "sg_e": [0],
    "rbc_normal": [1],  # Red Blood Cells (1 for normal)
    "pc_normal": [1],  # Pus Cells (1 for normal)
    "pcc_present": [0],  # Pus Cell Clumps (0 for not present)
    "ba_present": [0],  # Bacteria (0 for not present)
    "htn_yes": [0],  # Hypertension (0 for no)
    "dm_yes": [0],  # Diabetes Mellitus (0 for no)
    "cad_yes": [0],  # Coronary Artery Disease (0 for no)
    "appet_yes": [1],  # Appetite (1 for good)
    "pe_yes": [0],  # Pedal Edema (0 for no)
    "ane_yes": [1],  # Anemia (0 for no)
}

# Convert dictionary to DataFrame
test_df = pd.DataFrame(test_data_dict)

# Preprocess test data using the same scaler as in training (StandardScaler)
scaler = StandardScaler()

# Since we need to match the training data preprocessing, we scale the test data
test_input = scaler.fit_transform(test_df)

# Predict
prediction = load_best_model.predict(test_input)
print(prediction)

# Create a dictionary for the label decoding
label_map = {0: "no", 1: "yes"}

# Predict and map back to original labels
prediction_label = label_map.get(prediction[0], "Unknown")
print(f"The predicted classification is: {prediction_label}")

