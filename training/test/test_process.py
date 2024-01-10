import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from training.src.process import get_features, rename_columns

# Define a test suite using pytest_steps to organize and execute test steps.
# This suite consists of two steps: "get_features_step" and "rename_columns_step".
@test_steps("get_features_step", "rename_columns_step")
def test_processs_suite(test_step, steps_data):
    """
    Test suite to validate the 'get_features' and 'rename_columns' functions.

    Args:
        test_step (str): The current test step to execute.
        steps_data (object): An object to store data between steps.
    """
    if test_step == "get_features_step":
        get_features_step(steps_data)
    elif test_step == "rename_columns_step":
        rename_columns_step(steps_data)

def get_features_step(steps_data):
    """
    Test step to validate the 'get_features' function.

    Args:
        steps_data (object): An object to store data between steps.
    """
    # Create a sample DataFrame for testing.
    data = pd.DataFrame(
        {
            "Education": ["Bachelors", "Masters"],
            "City": ["Bangalore", "Prune"],
            "PaymentTier": [2, 3],
            "Age": [30, 21],
            "Gender": ["Male", "Female"],
            "EverBenched": ["No", "Yes"],
            "ExperienceInCurrentDomain": [2, 3],
            "LeaveOrNot": [0, 1],
        }
    )
    features = [
        "City",
        "PaymentTier",
        "Age",
        "Gender",
        "EverBenched",
        "ExperienceInCurrentDomain",
    ]
    target = "LeaveOrNot"

    # Call the 'get_features' function to obtain the target and features.
    y, X = get_features(target, features, data)

    # Define a schema for the expected structure of the processed DataFrame.
    schema = DataFrameSchema(
        {
            "City[Bangalore]": Column(float, Check.isin([0.0, 1.0])),
            "City[Prune]": Column(float, Check.isin([0.0, 1.0])),
            "Gender[T.Male]": Column(float, Check.isin([0.0, 1.0])),
            "EverBenched[T.Yes]": Column(float, Check.isin([0.0, 1.0])),
            "PaymentTier": Column(float, Check.isin([1, 2, 3])),
            "Age": Column(float, Check.greater_than(10)),
            "ExperienceInCurrentDomain": Column(
                float, Check.greater_than_or_equal_to(0)
            ),
        }
    )

    # Validate that the processed DataFrame adheres to the defined schema.
    schema.validate(X)

    # Store the processed DataFrame in the steps_data object for later use.
    steps_data.X = X

def rename_columns_step(steps_data):
    """
    Test step to validate the 'rename_columns' function.

    Args:
        steps_data (object): An object to store data between steps.
    """
    # Call the 'rename_columns' function on the processed DataFrame.
    processed_X = rename_columns(steps_data.X)

    # Assert that the column names match the expected names after renaming.
    assert list(processed_X.columns) == [
        "City_Bangalore",
        "City_Prune",
        "Gender_T.Male",
        "EverBenched_T.Yes",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain",
    ]


