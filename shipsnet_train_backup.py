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

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

import pandas as pd

device = torch.device("cuda")

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer.autoguide.initialization import init_to_median

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from torchvision.datasets import ImageFolder

from dotenv import load_dotenv
import requests
import os

class BayesShipsCNN(PyroModule):
    def __init__(
        self,
        num_classes=2,
        device= torch.device("cuda"),
        activation='relu',
        prior_dist='gaussian',
        mu=0.0,
        b=1.0,
        prior_params=None
    ):
        super().__init__()

        # Store device
        self.device = device

        # Activation setup: accept string or callable
        if isinstance(activation, str):
            act_map = {
                'relu': F.relu,
                'tanh': F.tanh,
                'sigmoid': F.sigmoid,
                'sinusoidal': torch.sin,
                'relu6': F.relu6,
                'leaky_relu': F.leaky_relu,
                'selu': F.selu,
                'wg':self.actWG,
                'rwg':self.actRWG,
            }
            try:
                self.activation_fn = act_map[activation]
            except KeyError:
                raise ValueError(f"Unsupported activation: {activation}")
        elif callable(activation):
            self.activation_fn = activation
        else:
            raise ValueError("activation must be a string or callable")

        # Prior distribution setup
        self.prior_dist = prior_dist
        default_params = {'mu': mu, 'b': b}
        params = default_params if prior_params is None else prior_params
        self.prior_mu = torch.tensor(params.get('mu', mu), device=device)
        self.prior_b  = torch.tensor(params.get('b', b), device=device)

        print(f"Using prior distribution: {self.prior_dist} with mu={self.prior_mu.item()} and b={self.prior_b.item()}")

        # Layer definitions with priors
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1.weight = PyroSample(self._make_prior([32, 3, 3, 3]))
        self.conv1.bias   = PyroSample(self._make_prior([32]))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2.weight = PyroSample(self._make_prior([64, 32, 3, 3]))
        self.conv2.bias   = PyroSample(self._make_prior([64]))

        self.conv3 = PyroModule[nn.Conv2d](64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3.weight = PyroSample(self._make_prior([128, 64, 3, 3]))
        self.conv3.bias   = PyroSample(self._make_prior([128]))

        # Pooling and global average pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = PyroModule[nn.Linear](128, 256)
        self.fc1.weight = PyroSample(self._make_prior([256, 128]))
        self.fc1.bias   = PyroSample(self._make_prior([256]))

        self.fc2 = PyroModule[nn.Linear](256, num_classes)
        self.fc2.weight = PyroSample(self._make_prior([num_classes, 256]))
        self.fc2.bias   = PyroSample(self._make_prior([num_classes]))

    def actWG(self, x, alpha=1.0):
        # Weight-gradient activation
        return x * torch.exp(-alpha * x ** 2)
    
    def actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
        # compare elementwise with zero
        return torch.max(torch.zeros_like(wg), wg)

    def _make_prior(self, shape):
        """
        Construct a prior distribution based on self.prior_dist and parameters.
        """
        if self.prior_dist == 'gaussian':
            base = dist.Normal(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'laplace':
            base = dist.Laplace(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'uniform':
            base = dist.Uniform(-self.prior_b, self.prior_b)
        else:
            raise ValueError(f"Unsupported prior distribution: {self.prior_dist}")
        return base.expand(shape).to_event(len(shape))

    def forward(self, x, y=None):
        x = self.activation_fn(self.conv1(x).to(self.device))
        x = self.pool(x)

        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        x = self.activation_fn(self.conv3(x))
        x = self.pool(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.activation_fn(self.fc1(x))
        logits = self.fc2(x)

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits



shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std = [0.1899, 0.1569, 0.1515]

old_mean = [0.3444, 0.3803, 0.4078]
old_std = [0.0914, 0.0651, 0.0552]


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

# training SVI function

import os
import torch
import pyro
from tqdm import tqdm
import numpy as np

def train_svi_with_stats(
    model,
    guide,
    svi,
    train_loader,
    device,
    num_epochs=10,
    save_epochs=None,
    save_dir='results',
    model_filename_pattern='model_{activation}_{prior}_epoch_{epoch}_{timestamp}.pth',
    guide_filename_pattern='guide_{activation}_{prior}_epoch_{epoch}_{timestamp}.pth',
    param_store_filename_pattern='param_store_{activation}_{prior}_epoch_{epoch}_{timestamp}.pkl',
    accuracies_filename_pattern='accuracy_results_{activation}_{prior}_{timestamp}.csv',
    losses_filename_pattern='losses_{activation}_{prior}_{timestamp}.csv',
    model_config_filename_pattern='config_{activation}_{prior}_{timestamp}.json'
):
    """
    Train the SVI model, track losses/accuracies, and
    save artifacts only when accuracy improves, naming files
    like `model_relu_gaussian_epoch_3.pth`.
    """
    
    # Pull names off the model if available, else fall back
    #act_name  = getattr(model, 'activation', getattr(model, 'activation_name', 'act'))
    act_name = model.activation_fn.__name__ if hasattr(model.activation_fn, '__name__') else str(model.activation_fn)
    prior_name = getattr(model, 'prior_dist', 'prior')
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(save_dir, exist_ok=True)
    save_epochs = set(save_epochs or range(1, num_epochs+1))

    pyro.clear_param_store()
    model.to(device)

    epoch_losses, epoch_accuracies, accuracy_epochs = [], [], []
    weight_stats = {'epochs': [], 'means': [], 'stds': []}
    bias_stats   = {'epochs': [], 'means': [], 'stds': []}
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            total_loss += svi.step(images, labels)
            batches += 1

        avg_loss = total_loss / batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch} - ELBO Loss: {avg_loss:.4f}")

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            model.eval(); guide.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in tqdm(train_loader, desc=f"Acc check epoch {epoch}"):
                    images, labels = images.to(device), labels.to(device)
                    trace = pyro.poutine.trace(guide).get_trace(images)
                    replayed = pyro.poutine.replay(model, trace=trace)
                    logits = replayed(images)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct/total
            epoch_accuracies.append(acc); accuracy_epochs.append(epoch)
            print(f"Epoch {epoch} - Train Acc: {acc*100:.2f}%")

            # record stats...
            w_means, w_stds, b_means, b_stds = [], [], [], []
            for name, param in pyro.get_param_store().items():
                if 'loc' in name:
                    w_means.append(param.mean().item()); w_stds.append(param.std().item())
                elif 'scale' in name:
                    b_means.append(param.mean().item()); b_stds.append(param.std().item())
            weight_stats['epochs'].append(epoch)
            weight_stats['means'].append(w_means)
            weight_stats['stds'].append(w_stds)
            bias_stats['epochs'].append(epoch)
            bias_stats['means'].append(b_means)
            bias_stats['stds'].append(b_stds)

            for name, param in pyro.get_param_store().items():
                if 'loc' in name or 'scale' in name:
                    print(f"{name}: {param.detach().cpu().numpy()}")

            # only save when accuracy improves
            if acc > best_acc:
                best_acc = acc
                fname_model = model_filename_pattern.format(activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)
                fname_guide = guide_filename_pattern.format(activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)
                fname_ps    = param_store_filename_pattern.format(activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)

                torch.save(model.state_dict(), os.path.join(save_dir, fname_model))
                torch.save(guide.state_dict(), os.path.join(save_dir, fname_guide))
                pyro.get_param_store().save(os.path.join(save_dir, fname_ps))
                print(f"  ↳ Saved: {fname_model}, {fname_guide}, {fname_ps}")

    # save losses per epoch in a csv file, with consistent file naming
    accuracies_df = pd.DataFrame({
        'epoch': accuracy_epochs,
        'accuracy': epoch_accuracies
    })
    accuracies_df.to_csv(os.path.join(save_dir,accuracies_filename_pattern.format(activation=act_name, prior=prior_name, timestamp=timestamp)), index=False)

    loss_df = pd.DataFrame({
        'epoch': list(range(1, epoch + 1)),
        'loss': epoch_losses
    })
    loss_df.to_csv(os.path.join(save_dir,losses_filename_pattern.format(activation=act_name, prior=prior_name, timestamp=timestamp)), index=False)
            
    # save model configuration in a json file
    config = {
        'activation': act_name,
        'prior': prior_name,
        'num_epochs': num_epochs,
        'best_accuracy_at_epoch': accuracy_epochs[np.argmax(epoch_accuracies)],
        'best_accuracy': best_acc,
        'batch_size': train_loader.batch_size,
        'train_size': len(train_loader.dataset),
        'prior_params': {
            'mu': model.prior_mu.item(),
            'b': model.prior_b.item()
        },
    }
    config_filename = model_config_filename_pattern.format(activation=act_name, prior=prior_name, timestamp=timestamp)

    with open(os.path.join(save_dir, config_filename), 'w') as f:
        import json
        json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_filename}")

    return epoch_losses, epoch_accuracies, accuracy_epochs, weight_stats, bias_stats, os.path.join(save_dir, fname_model), os.path.join(save_dir, fname_guide), os.path.join(save_dir, fname_ps), timestamp


def plot_training_results_with_stats(losses, accuracies, accuracy_epochs, weight_stats, bias_stats, act_name, prior_name, timestamp):
    """Plot training results with weight and bias statistics"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.grid(True)
    
    # Plot 2: Training Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(accuracy_epochs, accuracies, 'o-')
    plt.title('Training Accuracy (Every 10 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot 3: Weight Statistics Boxplot
    plt.subplot(2, 2, 3)
    weight_data = []
    weight_labels = []
    
    for i, epoch in enumerate(weight_stats['epochs']):
        # Combine means and stds for this epoch
        epoch_data = weight_stats['means'][i] + weight_stats['stds'][i]
        weight_data.append(epoch_data)
        weight_labels.append(f'Epoch {epoch}')
    
    if weight_data:
        bp1 = plt.boxplot(weight_data, labels=weight_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
    
    plt.title('LOC Statistics Distribution')
    plt.xlabel('Epoch')
    plt.ylabel('LOC Values')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Bias Statistics Boxplot
    plt.subplot(2, 2, 4)
    bias_data = []
    bias_labels = []
    
    for i, epoch in enumerate(bias_stats['epochs']):
        # Combine means and stds for this epoch
        epoch_data = bias_stats['means'][i] + bias_stats['stds'][i]
        bias_data.append(epoch_data)
        bias_labels.append(f'Epoch {epoch}')
    
    if bias_data:
        bp2 = plt.boxplot(bias_data, tick_labels=bias_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
    
    plt.title('SCALE Statistics Distribution')
    plt.xlabel('Epoch')
    plt.ylabel('SCALE Values')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results_GP_shipsnet_elbo3/bayesian_cnn_training_results_{act_name}_{prior_name}_{timestamp}.png')
    #plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix


def predict_data(model, loader_of_interest, num_samples=10):
    model.eval()
    guide.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader_of_interest, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            logits_mc = torch.zeros(num_samples, images.size(0), model.fc2.out_features).to(device)

            for i in range(num_samples):
                guide_trace = pyro.poutine.trace(guide).get_trace(images)
                replayed_model = pyro.poutine.replay(model, trace=guide_trace)
                logits = replayed_model(images)
                logits_mc[i] = logits

            avg_logits = logits_mc.mean(dim=0)
            predictions = torch.argmax(avg_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions

def save_predictions_to_csv(labels, predictions, filename='predictions.csv'):
    df = pd.DataFrame({'True Label': labels, 'Predicted Label': predictions})
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

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


from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    num_classes = 2
    
    act_fn_list = ['gaussian', 'laplace', 'uniform']
    prior_list = ['tanh','sigmoid','relu','sinusoidal','relu6','wg','rwg']
    b_list = [10.0, 1.0, 0.1]

    #count how many combinations we have
    total_combinations = len(act_fn_list) * len(prior_list) * len(b_list)
    print(f"Total combinations to run: {total_combinations}")

    experiment_number = 0

    #'rwg','wg',
    for activation_iter in act_fn_list:
        for prior_iter in prior_list:
            for b_iter in b_list:

                experiment_number += 1
                experiment_time_start = time.time()
                #send telegram message to announce the start of the experiment (x/total combinations)
                send_telegram_message(
                    title=f"Experiment {experiment_number}/{total_combinations}",
                    message=f"Running with activation={activation_iter}, prior={prior_iter}, b={b_iter}"
                )

                pyro.clear_param_store()

                # print log to annoounce what experiment is running
                print("==========================================")
                print(f"Running experiment with activation={prior_iter}, prior={activation_iter}, b={b_iter}")
                print("==========================================")
                bayesian_model = BayesShipsCNN(num_classes,
                        device,
                        activation=prior_iter,
                        prior_dist=activation_iter,
                        mu = 0.0,
                        b= b_iter,
                        #prior_params={'mu': 0.0, 'b': b_iter})
                        )
                
                # 1) construct your guide so its locs start at p(w).mean=0
                guide = AutoDiagonalNormal(
                    bayesian_model,
                    init_loc_fn=init_to_median(num_samples=1),   # all μ_q ← prior mean (0)
                    init_scale=0.1               # set initial σ_q=0.1
                )

                optimizer = Adam({"lr": 1e-3,
                                  "weight_decay": 1e-4,
                                  })  # Increased from 1e-4 to 1e-3, weight decay added
                svi = pyro.infer.SVI(model=bayesian_model,
                                    guide=guide,
                                    optim=optimizer,
                                    loss=pyro.infer.Trace_ELBO(num_particles=3,
                                                                )) #TODO

                pyro.clear_param_store()

                # Ensure model and guide are on the correct device
                bayesian_model.to(device)
                guide.to(device)

                train_loader, test_loader = load_data(batch_size=16)
                
                losses, accuracies, accuracy_epochs, weight_stats, bias_stats, best_model_path, best_guide_path, best_param_store_path, experiment_timestamp = train_svi_with_stats(
                bayesian_model,
                guide,
                svi,
                train_loader,
                device,
                num_epochs=100,
                save_epochs=None,
                save_dir='results_GP_shipsnet_elbo3')
                
                act_name = bayesian_model.activation_fn.__name__ if hasattr(bayesian_model.activation_fn, '__name__') else str(bayesian_model.activation_fn)
                prior_name = getattr(bayesian_model, 'prior_dist', 'prior')

                plot_training_results_with_stats(losses, accuracies, accuracy_epochs, weight_stats, bias_stats, act_name, prior_name, experiment_timestamp)

                all_labels, all_predictions = predict_data(bayesian_model, test_loader, num_samples=10)
                cm = confusion_matrix(all_labels, all_predictions)
                #print accuracy from confusion matrix
                accuracy = np.trace(cm) / np.sum(cm)
                print(f"Accuracy from confusion matrix: {accuracy * 100:.6f}%")

                experiment_time_finish = time.time()

                save_predictions_to_csv(all_labels, all_predictions, os.path.join('results_GP_shipsnet_elbo3', f'predictions_{act_name}_{prior_name}_{experiment_timestamp}_{accuracy * 100:.0f}.csv'))

                send_telegram_message(
                    title=f"Experiment {experiment_number}/{total_combinations} Finished",
                    message=f"Activation: {prior_iter}, Prior: {activation_iter}, b: {b_iter}\n"
                            f"Best Model Test Accuracy: {accuracy * 100:.2f}%\n"
                            f"Time taken: {experiment_time_finish - experiment_time_start:.2f} seconds"
                )

                