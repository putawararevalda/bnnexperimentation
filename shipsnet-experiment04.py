import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

import pickle
from tqdm import tqdm
import copy

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoNormal

import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix

from bitflip import bitflip_float32

from torchvision.datasets import ImageFolder

import os

import json

shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std = [0.1899, 0.1569, 0.1515]

def load_data(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=shipsnet_mean, 
                             std=shipsnet_std)
    ])

    #dataset = datasets.EuroSAT(root='./data', transform=transform, download=True)
    dataset = ImageFolder(
    root="data/shipsnet/foldered",
    transform=transform
    )
    torch.manual_seed(42)

    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    with open('datasplit/shipsnet_split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    # Add num_workers and pin_memory for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader

class BayesShipsCNN(PyroModule):
    def __init__(
        self,
        num_classes=2,   # now 2 for Categorical
        device=torch.device("cuda"),
        activation='relu',
        prior_dist='gaussian',
        mu=0.0,
        b=1.0,
        prior_params=None
    ):
        super().__init__()
        self.device = device

        # Activation setup
        if isinstance(activation, str):
            act_map = {
                'relu': F.relu,
                'tanh': torch.tanh,
                'sigmoid': torch.sigmoid,
                'sin': torch.sin,
                'relu6': F.relu6,
                'leaky_relu': F.leaky_relu,
                'selu': F.selu,
                'actWG': self.actWG,
                'actRWG': self.actRWG,
            }
            self.activation_fn = act_map[activation]
        elif callable(activation):
            self.activation_fn = activation
        else:
            raise ValueError("activation must be a string or callable")

        # Prior setup
        self.prior_dist = prior_dist
        params = {'mu': mu, 'b': b} if prior_params is None else prior_params
        self.prior_mu = torch.tensor(params['mu'], device=device, dtype=torch.float32)
        self.prior_b  = torch.tensor(params['b'], device=device, dtype=torch.float32)

        print(f"[INFO] Using prior: {self.prior_dist} (mu={self.prior_mu.item()}, b={self.prior_b.item()})")

        # Layers
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=3, padding=1)
        self.conv1.weight = PyroSample(self._make_prior([32, 3, 3, 3]))
        self.conv1.bias   = PyroSample(self._make_prior([32]))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=3, padding=1)
        self.conv2.weight = PyroSample(self._make_prior([64, 32, 3, 3]))
        self.conv2.bias   = PyroSample(self._make_prior([64]))

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(self._make_prior([num_classes, 64 * 16 * 16]))
        self.fc1.bias   = PyroSample(self._make_prior([num_classes]))

    def actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x ** 2)

    def actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
        return torch.max(torch.zeros_like(wg), wg)

    def _make_prior(self, shape):
        if self.prior_dist == 'gaussian':
            base = dist.Normal(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'laplace':
            base = dist.Laplace(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'uniform':
            base = dist.Uniform(-self.prior_b, self.prior_b)
        else:
            raise ValueError(f"Unsupported prior: {self.prior_dist}")
        return base.expand(shape).to_event(len(shape))

    def forward(self, x, y=None):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc1(x)  # shape [batch, 2]

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


def load_model(timestamp):
    config_path = os.path.join(search_dir, config_files[timestamp])
    guide_path = os.path.join(search_dir, guide_files[timestamp])
    model_path = os.path.join(search_dir, model_files[timestamp])
    param_path = os.path.join(search_dir, param_files[timestamp])

    print(f"Loading model with config_path: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = BayesShipsCNN(
        num_classes=num_classes,
        device=device,
        activation=config['activation'],
        prior_dist=config['prior'],
        mu=config['prior_params']['mu'],
        b=config['prior_params'],
        prior_params=config.get('prior_params', None)
    ).to(device)

    # Load the guide
    #guide = AutoDiagonalNormal(model)

    # Load the model state
    model.load_state_dict(torch.load(model_path))
    
    # Load the guide state
    #guide.load_state_dict(torch.load(guide_path))

    return model, param_path

class NewInjector:
    def __init__(self, trained_model, device, test_loader, num_samples):
        """
        Initializes SEU injector
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = trained_model.to(self.device)
        self.test_loader = test_loader
        self.trained_model.eval()
        self.num_samples = num_samples

        #self.guide = AutoDiagonalNormal(self.trained_model).to(self.device)
        self.guide = AutoNormal(bayesian_model, init_scale=0.05).to(device)
        pyro.get_param_store().clear()
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        initial_labels, initial_predictions, initial_logits, initial_probs = self.predict_data_probs(self.num_samples)
        self.initial_accuracy = self.return_accuracy(initial_labels, initial_predictions)
        self.initial_probs = np.array(initial_probs)

        print(f"Initial accuracy: {self.initial_accuracy}")

    def predict_data_probs(self, num_samples=10):
        all_labels = []
        all_predictions = []
        all_logits = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                logits_mc = torch.zeros(num_samples, images.size(0), self.trained_model.fc1.out_features).to(self.device)

                for i in range(num_samples):
                    guide_trace = pyro.poutine.trace(self.guide).get_trace(images)
                    replayed_model = pyro.poutine.replay(self.trained_model, trace=guide_trace)
                    logits = replayed_model(images)
                    logits_mc[i] = logits

                avg_logits = logits_mc.mean(dim=0)
                predictions = torch.argmax(avg_logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.extend(avg_logits.cpu().numpy())
                all_probs.extend(F.softmax(avg_logits, dim=1).cpu().numpy())

        return all_labels, all_predictions, all_logits, all_probs

    def return_accuracy(self, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, all_predictions)
        return np.trace(cm) / np.sum(cm)

    def compute_softmax_difference(self, before_probs, after_probs):
        before_probs = np.array(before_probs)
        after_probs = np.array(after_probs)
        diff = np.abs(before_probs - after_probs)
        return np.max(diff, axis=1).mean()

    def compute_difference(self, original_val, modified_val):
        return abs(original_val - modified_val)
    
    #AutoNormal.locs.conv1.weight
    def run_seu_autonormal(self, location_index, parameter_name, layer, layer_module, bit_i, num_samples):
        assert parameter_name in ["locs", "scales"], "Parameter name must be 'locs' or 'scales'."
        assert bit_i in range(0, 33), "Bit index must be between 0 and 32."

        param_store_name = f"AutoNormal.{parameter_name}.{layer}.{layer_module}"
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_store_name)
            new_param = param.clone()
            new_param = new_param.view(-1) #flatten new param
            original_val = new_param[location_index].cpu().item()
            seu_val = bitflip_float32(original_val, bit_i)
            abs_diff = self.compute_difference(original_val, seu_val)
            new_param[location_index] = seu_val
            # return new_param to original shape
            new_param = new_param.view(param.shape)
            pyro.get_param_store().__setitem__(param_store_name, new_param)

            print(f"Original value: {original_val}, SEU value: {seu_val}, Abs difference: {abs_diff}")

        self.guide = AutoNormal(self.trained_model, init_scale=0.05).to(self.device)

        try:
            after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
            accuracy_after = self.return_accuracy(after_labels, after_predictions)
            softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)
        except:
            accuracy_after = np.nan
            softmax_diff = np.nan

        print(f"Accuracy after SEU: {accuracy_after}")
        print("===================================")

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff
        }


    def run_seu_autodiagonal_normal(self, location_index, bit_i, parameter_name="loc", num_samples=10):
        assert parameter_name in ["loc", "scale"], "Parameter name must be 'loc' or 'scale'."
        assert bit_i in range(0, 33), "Bit index must be between 0 and 32."

        param_store_name = f"AutoDiagonalNormal.{parameter_name}"
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_store_name)
            new_param = param.clone()
            original_val = new_param[location_index].cpu().item()
            seu_val = bitflip_float32(original_val, bit_i)
            abs_diff = self.compute_difference(original_val, seu_val)
            new_param[location_index] = seu_val
            pyro.get_param_store().__setitem__(param_store_name, new_param)

            print(f"Original value: {original_val}, SEU value: {seu_val}, Abs difference: {abs_diff}")

        self.guide = AutoDiagonalNormal(self.trained_model).to(self.device)

        try:
            after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
            accuracy_after = self.return_accuracy(after_labels, after_predictions)
            softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)
        except:
            accuracy_after = np.nan
            softmax_diff = np.nan

        print(f"Accuracy after SEU: {accuracy_after}")
        print("===================================")

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff
        }

    def run_seu_autodiagonal_normal_multi(self, location_indices, bit_i, parameter_name="loc",
                                          attack_ratio=1.0, num_samples=10, seed=None):
        assert parameter_name in ["loc", "scale"], "Parameter name must be 'loc' or 'scale'."
        assert bit_i in range(0, 33), "Bit index must be between 0 and 32."
        assert 0.0 <= attack_ratio <= 1.0, "Attack ratio must be between 0.0 and 1.0."

        if isinstance(location_indices, int):
            location_indices = [location_indices]

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        num_attacks = max(1, int(len(location_indices) * attack_ratio))
        attack_locations = np.random.choice(location_indices, size=num_attacks, replace=False)
        param_store_name = f"AutoDiagonalNormal.{parameter_name}"
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        abs_differences = []

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_store_name)
            new_param = param.clone()

            #print(f"Attacking {num_attacks} out of {len(location_indices)} locations:")

            for location_index in attack_locations:
                original_val = new_param[location_index].cpu().item()
                seu_val = bitflip_float32(original_val, bit_i)
                abs_diff = self.compute_difference(original_val, seu_val)
                abs_differences.append(abs_diff)
                new_param[location_index] = seu_val
                print(f"  Location {location_index}: {original_val} -> {seu_val}, Log diff: {abs_diff}")

            pyro.get_param_store().__setitem__(param_store_name, new_param)

        self.guide = AutoDiagonalNormal(self.trained_model).to(self.device)

        try:
            after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
            accuracy_after = self.return_accuracy(after_labels, after_predictions)
            softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)
            mean_abs_diff = float(np.mean(abs_differences))
        except:
            accuracy_after = np.nan
            softmax_diff = np.nan
            mean_abs_diff = np.nan

        #print(f"Accuracy after SEU: {accuracy_after}")
        #print("===================================")

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "mean_abs_difference": mean_abs_diff
        }

def load_model_config(timestamp):
    config_path = os.path.join(search_dir, config_files[timestamp])

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    return model_config

if __name__ == "__main__":
    train_loader, test_loader = load_data(batch_size=16)
    device = torch.device("cuda")
    num_classes = 2
    

    search_dir = 'results_GP_shipsnet_newslate'
    #list all .json files in the directory
    all_files = [f for f in os.listdir(search_dir)]
    json_files = [f for f in os.listdir(search_dir) if f.endswith('.json')]

    # excluding the format, get the last 16 characters of each filename
    timestamps = [f[:-5][-16:] for f in json_files]
    print("Timestamps found count:", len(timestamps))

    # remove some timestamps that are not needed
    # those are the ones that are not in the shipsnet_seu_result directory, without the .csv extension
    excluded_timestamps = [f[:-4][-16:] for f in [f for f in os.listdir('shipsnet_newslate_seu_result') if f.endswith('.csv')]]
    
    timestamps = [ts for ts in timestamps if ts not in excluded_timestamps]
    #timestamps = timestamps[:1]
    print("After excluding, timestamps count:", len(timestamps))

    # for each timestamp, look for every other files in the directory that contains the timestamp

    config_files = {}
    guide_files = {}
    model_files = {}
    param_files = {}

    for timestamp in timestamps:
        config_files[timestamp] = [f for f in all_files if timestamp in f and f.endswith('.json')][0]
        guide_files[timestamp] = [f for f in all_files if timestamp in f and f.startswith('guide')][0]
        model_files[timestamp] = [f for f in all_files if timestamp in f and f.startswith('model')][0]
        param_files[timestamp] = [f for f in all_files if timestamp in f and f.startswith('param')][0]

    for ts_idx in range(len(timestamps)):
        # clear pyro's param store
        pyro.clear_param_store()
        
        bayesian_model, pyro_param_store_path = load_model(timestamps[ts_idx])

        newinj = NewInjector(trained_model=bayesian_model, device=device, test_loader=test_loader, num_samples=10)
        
        model_config = load_model_config(timestamps[ts_idx])

        ## MAIN LOOP CODE

        results_df = pd.DataFrame(columns=["activation_fn",
                                        "prior",
                                        "best_accuracy",
                                        "prior_mu",
                                            "prior_b",
                                            "param_type",
                                        "location_index",
                                        "location_layer",
                                        "location_module",
                                        "bit_index", 
                                        "initial_accuracy", 
                                        "accuracy_after_seu", 
                                        "accuracy_change", 
                                        "softmax_difference", 
                                        "mean_abs_difference",
                                        ])

        # get the initial accuracy from the newinj object
        initial_accuracy = newinj.initial_accuracy

        attack_locations = ["beginning", "end"]

        layer_list = []

        for name, value in pyro.get_param_store().items():
            layer_list.append(name.split('.')[2])  # Extract the layer name from the param store key

        bit_i = 2

        for attack_location in attack_locations:
            for layer_list_iter in layer_list:
                for module_iter in ["weight", "bias"]:
                
                    if attack_location == "beginning":
                        target_index = 0
                    elif attack_location == "end":
                        target_index = -1

                    #print(f"Running SEU on {layer}.{param_name} at index {target_index} with bit flip 0")

                    for bit_iter in [0, 1, 3, 6, 10, 15, 21]:
                        for parameter_name in ["locs", "scales"]:
                            print(f"Running SEU for {layer_list_iter}.{module_iter} bit index {bit_iter}  for parameter {parameter_name}")
                            #result = newinj.run_seu_autodiagonal_normal(location_index=target_index, bit_i=bit_iter, parameter_name=parameter_name, num_samples=10)
                            result = newinj.run_seu_autonormal(target_index, parameter_name, layer_list_iter, module_iter, bit_iter, num_samples=10)
                            # use concat to save the result to a dataframe
                            #print(initial_accuracy)
                            iter_df = pd.DataFrame({
                                "activation_fn": model_config['activation'],
                                "prior": model_config['prior'],
                                "best_accuracy": model_config['best_accuracy'],
                                "prior_mu": model_config['prior_params']['mu'],
                                "prior_b": model_config['prior_params']['b'],
                                "param_type": parameter_name,
                                "location_index": target_index,
                                "location_layer": layer_list_iter,
                                "location_module": module_iter,
                                "bit_index": bit_iter,
                                "initial_accuracy": initial_accuracy,
                                "accuracy_after_seu": initial_accuracy + result["accuracy_change"],
                                "accuracy_change": result["accuracy_change"],
                                "softmax_difference": result["softmax_difference"],
                                "mean_abs_difference": result["absolute_difference"]
                            }, index=[0])

                            results_df = pd.concat([results_df, iter_df], ignore_index=True)


        results_df.to_csv(f'shipsnet_newslate_seu_result/{timestamps[ts_idx]}.csv', index=False)
        print(f"Results saved for timestamp {timestamps[ts_idx]}")