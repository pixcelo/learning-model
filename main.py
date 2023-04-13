import os
import sys
import traceback

from data_collection import collect_data
from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model


def main():
    try:
        print("Starting data collection...")
        collect_data()
        print("Data collection completed.")

        print("Starting data preprocessing...")
        preprocess_data()
        print("Data preprocessing completed.")

        print("Starting model training...")
        train_model()
        print("Model training completed.")

        print("Starting model evaluation...")
        evaluate_model()
        print("Model evaluation completed.")

        print("All processes completed successfully.")

    except Exception as e:
        error_message = f"Error occurred: {e}\n"
        error_traceback = traceback.format_exc()
        print(error_message)
        print(error_traceback)


if __name__ == "__main__":
    main()
