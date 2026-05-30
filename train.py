"""train.py --- Main training script for the Boston housing price prediction model using Decision Tree Regressor.
"""

from sklearn.tree import DecisionTreeRegressor

from misc import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model
)


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = DecisionTreeRegressor(random_state=42)

    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    mse = evaluate_model(
        trained_model,
        X_test,
        y_test
    )

    print(f"Decision Tree MSE: {mse}")


if __name__ == "__main__":
    main()