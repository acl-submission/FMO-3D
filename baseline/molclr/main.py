import os
from pprint import pprint

import pandas as pd
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.dataset_test import SmilesToGraph
from models.ginet_finetune import GINet

def load_model(model_path, config_path):


    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset']['task'] = 'classification' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GINet(config['dataset']['task'], **config['model']).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device

def predict_aroma_probabilities(root_path, smiles_list, target_list):
    all_probabilities = {i:[] for i in ["nonStereoSMILES"]+target_list}
    all_probabilities["nonStereoSMILES"] = smiles_list

    for target in tqdm(target_list, desc=f"Models Predict"):
        print("Predicting", target)
        # Load the model for predicting animal aroma
        model_path = f"{root_path}/{target}/checkpoints/model.pth"
        config_path = f"{root_path}/{target}/checkpoints/config_finetune.yaml"

        if not os.path.exists(model_path):
            print(f"{target}The single data leads to no training！")
            del all_probabilities[target]
            continue

        model, device = load_model(model_path, config_path)

        # predict
        for smiles_string in smiles_list:
            transform = SmilesToGraph()
            data = transform(smiles_string)
            if data is None:
                print(f"SMILES The string is invalid: {smiles_string}")
                return None

            data_on_device = data.to(device)
            with torch.no_grad():
                __, pred = model(data_on_device)
                probabilities = F.softmax(pred, dim=-1)
                probability=probabilities[0, 1].item()

            all_probabilities[target].append(probability)
    return all_probabilities

def main():
    model_root_path = "first_finetune"
    predict_xlsx_path = "xlsx/openpom_test_dataset_primary.csv"
    save_path = f"output/primary_prediction_result.csv"

    # model_root_path = "second_finetune"
    # predict_xlsx_path = "xlsx/openpom_test_dataset_secondary.csv"
    # save_path = f"output/second_prediction_result.csv"

    # model_root_path = "third_finetune"
    # predict_xlsx_path = "xlsx/openpom_test_dataset_third.csv"
    # save_path = f"output/third_prediction_result.csv"

    df = pd.read_csv(predict_xlsx_path)
    smiles_list = df["smiles"]
    target_list = list(df.columns)[2:]

    result = predict_aroma_probabilities(model_root_path, smiles_list, target_list)
    # Handle the table header
    result=pd.DataFrame(result)
    result.columns = ["nonStereoSMILES"] + [i+"_probability" for i in result.columns[1:]]
    # Add the results of column 01
    for column in result.columns[1:]:
        new_column = column.split("_")[0] + "_prediction"
        # Simplify the code using list inference
        result[new_column] = [1 if i >= 0.5 else 0 for i in result[column]]
    # Save
    result.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()
