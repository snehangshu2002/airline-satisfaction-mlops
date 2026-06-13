import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame(
        {
        "ID": [1, 2, 3, 4],
        "Flight Distance": [1000, "2000", None, 1500],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Customer Type": ["Loyal Customer", "disloyal Customer", "Loyal Customer", "disloyal Customer"],
        "Type of Travel": ["Business travel", "Personal Travel", "Business travel", "Personal Travel"],
        "Class": ["Business", "Eco", "Eco Plus", "Business"],
        "Departure and Arrival Time Convenience": [1, 2, 3, 4],
        "Ease of Online Booking": [2, 3, 4, 5],
        "Check-in Service": [1, 2, 3, 4],
        "Online Boarding": [2, 3, 4, 5],
        "Gate Location": [1, 2, 3, 4],
        "On-board Service": [2, 3, 4, 5],
        "Seat Comfort": [1, 2, 3, 4],
        "Leg Room Service": [2, 3, 4, 5],
        "Cleanliness": [1, 2, 3, 4],
        "Food and Drink": [2, 3, 4, 5],
        "In-flight Service": [1, 2, 3, 4],
        "In-flight Wifi Service": [2, 3, 4, 5],
        "In-flight Entertainment": [1, 2, 3, 4],
        "Baggage Handling": [2, 3, 4, 5],
        "Satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied", "neutral or dissatisfied"],
        }
    )

@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "Flight Distance": [1000, 2000, 1500, 1800],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Customer Type": ["Loyal Customer", "disloyal Customer", "Loyal Customer", "disloyal Customer"],
        "Type of Travel": ["Business travel", "Personal Travel", "Business travel", "Personal Travel"],
        "Class": ["Business", "Eco", "Eco Plus", "Business"],
        "Departure and Arrival Time Convenience": [1, 2, 3, 4],
        "Ease of Online Booking": [2, 3, 4, 5],
        "Check-in Service": [1, 2, 3, 4],
        "Online Boarding": [2, 3, 4, 5],
        "Gate Location": [1, 2, 3, 4],
        "On-board Service": [2, 3, 4, 5],
        "Seat Comfort": [1, 2, 3, 4],
        "Leg Room Service": [2, 3, 4, 5],
        "Cleanliness": [1, 2, 3, 4],
        "Food and Drink": [2, 3, 4, 5],
        "In-flight Service": [1, 2, 3, 4],
        "In-flight Wifi Service": [2, 3, 4, 5],
        "In-flight Entertainment": [1, 2, 3, 4],
        "Baggage Handling": [2, 3, 4, 5],
        "Satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied", "neutral or dissatisfied"],
    })

@pytest.fixture
def sample_test_df():
    return pd.DataFrame({
        "Flight Distance": [1200, 1600],
        "Gender": ["Female", "Male"],
        "Customer Type": ["Loyal Customer", "disloyal Customer"],
        "Type of Travel": ["Business travel", "Personal Travel"],
        "Class": ["Business", "Eco"],
        "Departure and Arrival Time Convenience": [2, 3],
        "Ease of Online Booking": [3, 4],
        "Check-in Service": [2, 3],
        "Online Boarding": [3, 4],
        "Gate Location": [2, 3],
        "On-board Service": [3, 4],
        "Seat Comfort": [2, 3],
        "Leg Room Service": [3, 4],
        "Cleanliness": [2, 3],
        "Food and Drink": [3, 4],
        "In-flight Service": [2, 3],
        "In-flight Wifi Service": [3, 4],
        "In-flight Entertainment": [2, 3],
        "Baggage Handling": [3, 4],
        "Satisfaction": ["satisfied", "neutral or dissatisfied"],
    })



@pytest.fixture
def dummy_model():
    class DummyModel:
        feature_importances_ = [0.2, 0.1, 0.7]

        def predict(self, X):
            return [0 for _ in range(len(X))]

        def predict_proba(self, X):
            return [[0.2, 0.8] for _ in range(len(X))]

    return DummyModel()
