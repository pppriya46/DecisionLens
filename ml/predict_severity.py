# ml/predict_severity.py


import os
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "ml/models/severity_rf_v1.pkl"
ENCODER_PATH = "ml/models/label_encoders.pkl"


def load_model():
    """Load trained model and encoders from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run ml/severity_model.py first."
        )
    
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders



def predict_severity(
    category: str,
    subcategory: str,
    contact_type: str,
    reassignment_count: int = 0,
    reopen_count: int = 0,
    sys_mod_count: int = 0,
    made_sla: bool = True,
    knowledge: bool = False,
    opened_at=None,
) -> dict:
    """
    Predict the priority of a new incident.
    
    Returns a dict with:
    - predicted_priority: the predicted class (e.g. '1 - Critical')
    - confidence: probability of the predicted class (0.0 to 1.0)
    - all_probabilities: probability for each class
    """
    model, encoders = load_model()


    # Time features
    if opened_at is not None:
        opened_at = pd.to_datetime(opened_at, errors='coerce')
        hour_of_day = opened_at.hour if pd.notna(opened_at) else 12
        day_of_week = opened_at.dayofweek if pd.notna(opened_at) else 0
    else:
        import datetime
        now = datetime.datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()

    is_weekend = int(day_of_week >= 5)
    is_business_hours = int(9 <= hour_of_day <= 17)


    def safe_encode(encoder, value):
        """Encode a value, returning -1 if unseen."""
        classes = list(encoder.classes_)
        return classes.index(value) if value in classes else -1

    category_encoded = safe_encode(
        encoders['category'], str(category).strip()
    )
    subcategory_encoded = safe_encode(
        encoders['subcategory'], str(subcategory).strip()
    )
    contact_type_encoded = safe_encode(
        encoders['contact_type'], str(contact_type).strip()
    )

    features = pd.DataFrame([{
        'category_encoded':      category_encoded,
        'subcategory_encoded':   subcategory_encoded,
        'contact_type_encoded':  contact_type_encoded,
        'reassignment_count':    int(reassignment_count),
        'reopen_count':          int(reopen_count),
        'sys_mod_count':         int(sys_mod_count),
        'made_sla':              int(made_sla),
        'knowledge':             int(knowledge),
        'hour_of_day':           hour_of_day,
        'day_of_week':           day_of_week,
        'is_weekend':            is_weekend,
        'is_business_hours':     is_business_hours,
    }])

    predicted_index = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_priority = encoders['priority'].classes_[predicted_index]
    confidence = float(probabilities[predicted_index])

    all_probabilities = {
        cls: float(prob)
        for cls, prob in zip(encoders['priority'].classes_, probabilities)
    }

    return {
        "predicted_priority": predicted_priority,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probabilities,
    }



if __name__ == "__main__":
    print("Testing severity prediction...\n")

    # Test case 1 — should be high/critical
    result1 = predict_severity(
        category="Software",
        subcategory="Email",
        contact_type="Phone",
        reassignment_count=3,
        reopen_count=2,
        sys_mod_count=10,
        made_sla=False,
        knowledge=False,
    )
    print("Test 1 — High reassignment, SLA breached:")
    print(f"  Predicted: {result1['predicted_priority']}")
    print(f"  Confidence: {result1['confidence']:.1%}")
    print(f"  All probabilities: {result1['all_probabilities']}\n")

    # Test case 2 — should be low/moderate
    result2 = predict_severity(
        category="Hardware",
        subcategory="Laptop",
        contact_type="Self service",
        reassignment_count=0,
        reopen_count=0,
        sys_mod_count=1,
        made_sla=True,
        knowledge=True,
    )
    print("Test 2 — Simple hardware request, SLA met:")
    print(f"  Predicted: {result2['predicted_priority']}")
    print(f"  Confidence: {result2['confidence']:.1%}")
    print(f"  All probabilities: {result2['all_probabilities']}\n")

    # Test case 3 — network outage scenario
    result3 = predict_severity(
        category="Network",
        subcategory="Wireless",
        contact_type="Phone",
        reassignment_count=5,
        reopen_count=3,
        sys_mod_count=20,
        made_sla=False,
        knowledge=False,
    )
    print("Test 3 — Network outage, multiple reassignments:")
    print(f"  Predicted: {result3['predicted_priority']}")
    print(f"  Confidence: {result3['confidence']:.1%}")
    print(f"  All probabilities: {result3['all_probabilities']}\n")