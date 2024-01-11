import joblib
import bentoml

def save_to_bentoml():
    model_path = "/home/nikhil/Projects/MLOPS/models/xgboost"
    model = joblib.load(model_path)  # Load the XGBoost model

    # Now save the model object
    bento_model = bentoml.xgboost.save_model("xgboost", model)


if __name__ == "__main__":
    save_to_bentoml()
