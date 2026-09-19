import pickle

with open("model.bin", "rb") as f:
    pipeline = pickle.load(f)


def predict_single(customer):
    result = pipeline.predict_proba([customer])[0, 1]
    return result

def lambda_handler(event, context):
    print("Params:", event)
    customer = event["customer"]
    prob = predict_single(customer)
    return {
        "churn_prob": float(prob),
        "churn": bool(prob >= 0.5)
    }