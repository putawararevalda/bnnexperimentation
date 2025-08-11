import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch

import pickle
import numpy as np

from bitflip import bitflip_float32

from torchvision.datasets import ImageFolder

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import json

import argparse

from dotenv import load_dotenv
import requests

import pandas as pd

parser = argparse.ArgumentParser(description='Inject SEU to a Deterministic Model')

parser.add_argument('--save-dir', type=str, nargs='?', action='store', default='shipsnet_newslate_guide_seu_result_experiment',
                help='Where to save the file. Default: \'shipsnet_newslate_guide_seu_result_experiment\'.')
parser.add_argument('--search-dir', type=str, nargs='?', action='store', default='results_GP_shipsnet_newslate_guide',
                help='Where are the files to load. Default: \'results_GP_shipsnet_newslate_guide\'.')
parser.add_argument('--model-variant', type=str, nargs='?', action='store', default='00',
            help='Model to run. Default: \'00\'.')

args = parser.parse_args()

def send_telegram_message(title, message):
    load_dotenv('.env')
    token = os.getenv('TELEGRAM_BOT_TOKEN')

    try:
        response = requests.post(f'https://api.telegram.org/bot{token}/sendMessage', data={
            'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'text': f'{title}\n{message}',
            #'parse_mode': 'Markdown'
        })
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")
        return None

dataset = ImageFolder(
    root="data/shipsnet/foldered",
    transform=transforms.ToTensor()
)

device = torch.device("cuda")

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=1)

shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std = [0.1899, 0.1569, 0.1515]

def load_data_withval(batch_size=16, val_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=shipsnet_mean, std=shipsnet_std)
    ])

    dataset = ImageFolder(
        root="data/shipsnet/foldered",
        transform=transform
    )

    torch.manual_seed(42)

    with open('datasplit/shipsnet_split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        full_train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    # Split the full train dataset into train and val
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = load_data_withval(16)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartPool(nn.Module):
    """
    A “smart” max‐pool that detects outliers (values > threshold) and, if desired,
    replaces them with the 2nd‐largest value in the window.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 10.0,
        detect_only: bool = False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.detect_only = detect_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        ks, st = self.kernel_size, self.stride

        # Unfold into patches of shape (N, C, ks*ks, L) where L = #windows per image
        patches = F.unfold(x, kernel_size=ks, stride=st)  # → (N, C*ks*ks, L)
        patches = patches.view(N, C, ks*ks, -1)           # → (N, C, ks*ks, L)

        # Find top‐2 values in each patch
        top2_vals, _ = torch.topk(patches, 2, dim=2)      # → (N, C, 2, L)
        max1 = top2_vals[:, :, 0, :]                      # → (N, C, L)
        max2 = top2_vals[:, :, 1, :]                      # → (N, C, L)

        # Detect any spikes above threshold
        spikes = max1 > self.threshold                    # → (N, C, L)
        #if spikes.any():
        #    warnings.warn(f"SmartPool: detected {int(spikes.sum())} pooled values above threshold={self.threshold}")

        # If correction is off, just return the regular max‐pooled result
        if self.detect_only:
            out = max1
        else:
            # Replace each spike with the 2nd‐largest value
            out = torch.where(spikes, max2, max1)

        # Fold back to (N, C, H_out, W_out)
        H_out, W_out = (H // ks, W // ks)
        out = out.view(N, C * 1, -1)                      # → (N, C, L)
        out = out.view(N, C, H_out, W_out)
        return out

class ShipsCNNCustom(nn.Module):
    def __init__(self, num_classes=2, 
                 activation='relu',
                 smartpool_switch=False,
                 pool_threshold=10.0,
                 pool_detect_only=False,
                 dropout_switch=False,
                 dropout_p=0.5):
        super().__init__()

        # Activation setup (same as BayesShipsCNN)
        act_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'sin': torch.sin,
            'relu6': F.relu6,
            #'leaky_relu': F.leaky_relu,
            #'selu': F.selu,
            'actWG': self.actWG,
            'actRWG': self.actRWG,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation_fn = act_map[activation]

        # Layers: Same as BayesShipsCNN (2 conv layers + pooling)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # SmartPool or standard MaxPool
        if smartpool_switch:
            self.pool = SmartPool(
                kernel_size=2,
                stride=2,
                threshold=pool_threshold,
                detect_only=pool_detect_only
            )
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Dropout
        self.dropout_switch = dropout_switch
        if dropout_switch:
            self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layer
        # BayesShipsCNN flattens [B,64,16,16] → fc1: 64*16*16 → 2
        self.fc1 = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        if self.dropout_switch and self.training:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        return logits

    def actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x ** 2)

    def actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
        return torch.max(torch.zeros_like(wg), wg)
    
def get_last_16_chars_of_pth_files(directory):
    pth_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    last_16_chars = [f[-20:-4] for f in pth_files]  # Exclude the '.pth' extension
    return last_16_chars

class DeterministicInjector:
    def __init__(self, trained_model, device, test_loader):
        """
        Initializes SEU injector
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = trained_model.to(self.device)
        self.test_loader = test_loader
        self.trained_model.eval()
        self.criterion = nn.CrossEntropyLoss()

        initial_labels, initial_predictions, initial_logits, initial_probs = self.predict_data_probs()
        self.initial_accuracy = self.return_accuracy(initial_labels, initial_predictions)

        self.initial_probs = np.array(initial_probs)

        print(f"Initial accuracy: {self.initial_accuracy}")

    def predict_data_probs(self):
        all_labels = []
        all_predictions = []
        all_logits = []
        all_probs = []

        with torch.no_grad():
            #for imgs, labels in val_loader:
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs= self.trained_model(images)
                loss   = self.criterion(outputs, labels)
                preds  = outputs.argmax(dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

        return all_labels, all_predictions, all_logits, all_probs

    def return_accuracy(self, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, all_predictions)
        return np.trace(cm) / np.sum(cm)

    def compute_softmax_difference(self, before_probs, after_logits, penalty=1.0):
        """
        before_probs: list or array, shape (N, C), all finite probabilities
        after_logits: list or array, shape (N, C), raw logits (may contain ±inf)
        penalty: float, the per‑example penalty to use if logits are nonfinite
        
        Returns the mean over N examples of either
        - max_i |before_probs[n,i] − after_probs[n,i]|,  if after_logits[n] is finite
        - penalty,                                    otherwise
        """
        before = np.asarray(before_probs, dtype=np.float32)
        after_logits = torch.from_numpy(np.asarray(after_logits, dtype=np.float32))
        N, C = after_logits.shape

        # 1) detect which rows of after_logits are all finite
        finite_mask = torch.isfinite(after_logits).all(dim=1).numpy()  # shape (N,)

        # 2) safe‑softmax only on the finite ones
        safe_after_probs = torch.zeros_like(after_logits)
        if finite_mask.any():
            good_logits = after_logits[finite_mask]
            # (you can optionally do the “stable” shift here)
            safe_after_probs[finite_mask] = F.softmax(good_logits, dim=1)
        safe_after_probs = safe_after_probs.numpy()

        # 3) compute per‑example diff, using the penalty where needed
        diffs = np.empty(N, dtype=np.float32)
        for n in range(N):
            if not finite_mask[n]:
                diffs[n] = penalty
            else:
                diffs[n] = np.max(np.abs(before[n] - safe_after_probs[n]))
        return diffs.mean()

    def compute_difference(self, original_val, modified_val):
        return abs(original_val - modified_val)

    def run_seu(
        self,
        location_index: int,
        #param_unique: str, for bayesian : AutoX
        #parameter_name: str, for bayesian : locs, scales, lows, widths, etc.
        layer: str, #conv1, conv2, fc1
        layer_module: str, #weight, bias
        bit_i: int,
        ):

        assert 0 <= bit_i < 32, "Bit index must be between 0 and 31."

        layer_name__ = f"{layer}.{layer_module}"

        remarks = ""

        self.trained_model.eval()

        with torch.no_grad():
            for layer_name, tensor in self.trained_model.named_parameters():
                if layer_name == layer_name__:  # match exact target
                    break
            else:
                raise ValueError(f"Layer {layer_name__} not found in the model.")
        
        # 1) Grab the original tensor and flatten it
        param = tensor.data.clone()  # copy original tensor values
        flat  = param.clone().view(-1)

        # return to my SEU code style
        # 2) Extract, flip one bit, and wrap back as a tensor
        orig_val = flat[location_index].cpu().item()
        flipped  = bitflip_float32(orig_val, bit_i)      # Python float

        seu_val  = torch.tensor(
                flipped, dtype=param.dtype, device=param.device
            )

        abs_diff = self.compute_difference(orig_val, flipped)

        # 3) Replace the value in the original tensor
        tensor.data.view(-1)[location_index] = seu_val

        print(
                f"Parameter '{layer_name__}[{location_index}]': "
                f"{orig_val:.6g} → {flipped:.6g}  "
                f"(abs diff {abs_diff:.3g})"
            )

        # 4) Recompute the model's output
        after_labels, after_preds, after_logits, after_probs = self.predict_data_probs()
        accuracy_after = self.return_accuracy(after_labels, after_preds)
        softmax_diff  = self.compute_softmax_difference(self.initial_probs, after_probs)

        # 5) Restore the original tensor value
        tensor.data.view(-1)[location_index] = orig_val

        print(f"Accuracy after SEU: {accuracy_after:.3%}")
        print("===================================")

        return {
            "accuracy_change":    accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff,
            "remarks": remarks
        }

if __name__ == "__main__":

    #directory_target = "results_shipsnet_deterministic_00"
    directory_target = args.search_dir
    timestamps = get_last_16_chars_of_pth_files(directory_target)

    experiment_iteration = 0

    #timestamps = ['20250808_202316'] #01
    #timestamps = ['20250808_202843'] #02
    timestamps = ['20250808_203118'] #03

    for ts_idx in range(len(timestamps)):
        experiment_iteration += 1

        send_telegram_message(title="DETERMINISTIC ShipsNet SEU Experiment Progress",
                message=f"Running experiment {experiment_iteration}/{len(timestamps)} for timestamp {timestamps[ts_idx]}. Save directory: {args.save_dir}."
            )
        
        #LOAD MODEL
        pth_files = [f for f in os.listdir(directory_target) if f.endswith('.pth')]
        for pth_file in pth_files:
            if timestamps[ts_idx] in pth_file:
                model_path = os.path.join(directory_target, pth_file)
                print(f"Loading model from {model_path}")
                # split the filename by the / first
                timestamp_activation = model_path.split('\\')[-1].split('_')[2]

                print(model_path)

                model_variant = args.model_variant

                if model_variant == "00":
                    model = ShipsCNNCustom(activation=timestamp_activation).to(device)
                elif model_variant == "01":
                    model = ShipsCNNCustom(activation=timestamp_activation,
                                        smartpool_switch=True).to(device)
                elif model_variant == "02":
                    model = ShipsCNNCustom(activation=timestamp_activation,
                                        dropout_switch=True).to(device)
                elif model_variant == "03":
                    model = ShipsCNNCustom(activation=timestamp_activation).to(device)
                else:
                    raise ValueError(f"Unsupported model variant: {model_variant}")
                #model = ShipsCNN(num_classes=2, activation=model_path.split('\\')[-1].split('_')[2])
                model.load_state_dict(torch.load(model_path))
                model.eval()
                break

        results_df = pd.DataFrame(columns=[
                                    "activation_fn",
                                    #"prior",
                                    #"best_accuracy",
                                    #"prior_mu",
                                    #"prior_b",
                                    #"param_type",
                                    "location_index",
                                    "location_layer",
                                    "location_module",
                                    "bit_index", 
                                    "initial_accuracy", 
                                    "accuracy_after_seu", 
                                    "accuracy_change", 
                                    "softmax_difference", 
                                    "mean_abs_difference",
                                    "remarks"
                                    ])

        #LOAD INJECTION OBJECT
        DeterministicInjection = DeterministicInjector(trained_model=model, device=device, test_loader=test_loader)

        initial_accuracy = DeterministicInjection.initial_accuracy

        attack_locations = ["beginning", "end"]
        layer_list = ["conv1", "conv2", "fc1"] 

        # THE LOOP STARTS HERE
        for attack_location in attack_locations:
            for layer_list_iter in layer_list:
                for module_iter in ["weight", "bias"]:
                
                    if attack_location == "beginning":
                        target_index = 0
                    elif attack_location == "end":
                        target_index = -1

                    for bit_iter in [0, 1, 3, 10]:

                        print(f"Running SEU for {layer_list_iter}.{module_iter} bit index {bit_iter}")

                        result = DeterministicInjection.run_seu(
                            location_index=target_index,
                            layer=layer_list_iter,
                            layer_module=module_iter,
                            bit_i=bit_iter
                        )

                        iter_df = pd.DataFrame({
                            "activation_fn": model.activation_fn.__name__,
                            #"prior": model.prior,
                            #"best_accuracy": model.best_accuracy,
                            #"prior_mu": model.prior_mu,
                            #"prior_b": model.prior_b,
                            #"param_type": model.param_type,
                            "location_index": target_index,
                            "location_layer": layer_list_iter,
                            "location_module": module_iter,
                            "bit_index": bit_iter, 
                            "initial_accuracy": initial_accuracy, 
                            "accuracy_after_seu": initial_accuracy + result["accuracy_change"], 
                            "accuracy_change": result["accuracy_change"], 
                            "softmax_difference": result["softmax_difference"], 
                            "mean_abs_difference": result["absolute_difference"],
                            "remarks": result["remarks"]
                        }, index=[0])

                        results_df = pd.concat([results_df, iter_df], ignore_index=True)

                        
        results_df.to_csv(os.path.join(args.save_dir,f'{timestamps[ts_idx]}.csv'), index=False)
        print(f"Results saved for timestamp {timestamps[ts_idx]}")

        send_telegram_message(title="DETERMINISTIC ShipsNet SEU Experiment",
            message=f"Finished experiment {experiment_iteration}/{len(timestamps)}. Save directory: {args.save_dir}. Timestamp: {timestamps[ts_idx]}.")

    