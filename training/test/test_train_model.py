import joblib
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ModelErrorAnalysis
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath

from training.src.train_model import load_data


def test_xgboost():
    """
    Test the XGBoost model using the deepchecks library for model error analysis.

    This function loads the XGBoost model, testing data, and performs model error analysis
    using deepchecks to identify potential issues in the model's predictions.

    Note: Ensure that the necessary configurations are specified in the Hydra configuration files.

    Returns:
        None
    """
    # Initialize Hydra and load configuration settings
    with initialize(version_base=None, config_path="../../config"):
        config = compose(config_name="main")

    # Load the trained XGBoost model
    model_path = abspath(config.model.path)
    model = joblib.load(model_path)

    # Load training and testing data
    X_train, X_test, y_train, y_test = load_data(config.processed)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Create deepchecks Dataset objects for training and testing
    train_ds = Dataset(train_df, label="LeaveOrNot")
    validation_ds = Dataset(test_df, label="LeaveOrNot")

    # Define a model error analysis check with a specified minimum error model score
    check = ModelErrorAnalysis(min_error_model_score=0.3)

    # Run the model error analysis check on the training and testing datasets with the model
    check.run(train_ds, validation_ds, model)


# Main entry point to run the test
if __name__ == "__main__":
    test_xgboost()
