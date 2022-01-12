from src.utils.all_utils import read_yaml, create_directory, save_local_df, save_reports
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2

def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]

    test_data_filename = config["artifacts"]["test"]
    test_data_path = os.path.join(artifacts_dir,split_data_dir, test_data_filename)
    test_data =pd.read_csv(test_data_path)

    test_y = test_data['quality']
    test_x = test_data.drop("quality", axis=1)

    model_dir =  config["artifacts"]["model_dir"]
    model_file_name =  config["artifacts"]["model_file"]
    model_path = os.path.join(artifacts_dir, model_dir, model_file_name)

    lr = joblib.load(model_path)
    
    predicted_values = lr.predict(test_x)
    
    rmse, mae, r2 = evaluate_metrics(test_y, predicted_values)

    scores_dir = config["artifacts"]["reports_dir"]
    scores_file_name = config["artifacts"]["scores"]

    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    create_directory([scores_dir_path])

    scores_file_path = os.path.join(scores_dir_path, scores_file_name)

    scores = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    save_reports(report=scores,report_path= scores_file_path)





if __name__ == '__main__':
    args =argparse.ArgumentParser()

    args.add_argument("--config", "-c", default = "config/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")

    parsed_args = args.parse_args()
    evaluate(config_path= parsed_args.config, params_path=parsed_args.params)