import pickle

def main():
    print("Hello from homework!")
    model_file_path = "pipeline_v1.bin"
    with open(model_file_path, "rb") as model_file:
        dv, model = pickle.load(model_file)

    X = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
    X = dv.transform(X)
    predictions = model.predict_proba(X)[0,1]

    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
