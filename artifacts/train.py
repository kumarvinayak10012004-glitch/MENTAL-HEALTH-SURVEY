import json
import joblib
import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.model import MLP

ARTIFACTS_DIR = "artifacts"
SCHEMA_PATH = f"{ARTIFACTS_DIR}/schema.json"
PREPROCESSOR_PATH = f"{ARTIFACTS_DIR}/preprocessor.joblib"
MODEL_PATH = f"{ARTIFACTS_DIR}/model.pth"

df = pd.read_csv("train.csv")

numeric_cols = ["Age", "Work_Hours", "Sleep_Hours"]
categorical_cols = ["Gender", "Job_Role", "Family_History"]
target_col = "Depression"

schema = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "feature_cols": numeric_cols + categorical_cols,
    "target_col": target_col
}

with open(SCHEMA_PATH, "w") as f:
    json.dump(schema, f, indent=2)

X = df[schema["feature_cols"]]
y = df[target_col]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

joblib.dump(preprocessor, PREPROCESSOR_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

model = MLP(input_dim=X_processed.shape[1])
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

for epoch in range(20):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

torch.save(
    {
        "input_dim": X_processed.shape[1],
        "model_state": model.state_dict()
    },
    MODEL_PATH
)

print("✅ Training complete & artifacts saved")
