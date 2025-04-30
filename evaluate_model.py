import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from networks.Connect4Net import Connect4Net
from util.data_transformer import DataTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [
    "connect4_4x4_alpha_zero_50k.pt",
    "connect4_4x4_supervised.pt",
    "connect4_4x4_supervised_100k.pt",
    "connect4_4x4_supervised_50k.pt",
]



############################### Evaluate ###############################
def evaluate_soft(model, loader, threshold=0.01):
    model.eval()
    with torch.no_grad():
        total = 0
        good_enough = 0
        for boards, target_policy, _ in loader:
            logits, _ = model(boards)
            probs = torch.softmax(logits, dim=1)  # [B, 4]

            # Check if predicted column is within threshold of max target value
            for i in range(probs.size(0)):
                pred_col = torch.argmax(probs[i])
                target_probs = target_policy[i]
                max_val = torch.max(target_probs)
                if target_probs[pred_col] >= (max_val - threshold):
                    good_enough += 1
                total += 1
        print(
            f"Soft Policy Accuracy (within {threshold:.2f} of max): {good_enough}/{total} = {good_enough/total:.2%}"
        )


data_path = "data/connect4_4x4_training_data_50k.npy"
data_transformer = DataTransformer(data_path)
train_loader = data_transformer.get_training_data()
test_loader = data_transformer.get_testing_data()

directory = "models"

for model_name in models:
    model_path = os.path.join(directory, model_name)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist. Skipping.")
        continue

    print(f"Evaluating model: {model_name}")
    model = Connect4Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    evaluate_soft(model, test_loader, threshold=0.01)



# model_path = "models/connect4_4x4_supervised_100k.pt"
# model = Connect4Net().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# evaluate_soft(model, test_loader, threshold=0.01)