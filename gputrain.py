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

from tqdm import tqdm

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.model import BayesianCNNSingleFCCustomWGBN

# for telegram notifications
from dotenv import load_dotenv
import requests

from utils.model import BayesianCNNSingleFCCustom, LaplaceBayesianCNNSingleFCCustom
import pandas as pd
import json

#===============================================================

parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on EuroSAT with Variational Inference')

parser.add_argument('--epochs', type=int, nargs='?', action='store', default=10,
                    help='How many epochs to train. Default: 10.')
parser.add_argument('--variant', type=str, nargs='?', action='store', default='Base',
                    help='Model variant to choose from. Options are \'Base\', \'Adaptive\'. Default: \'Base\'')
parser.add_argument('--prior', type=str, nargs='?', action='store', default='Gaussian_prior',
                    help='Model prior to choose from. Options are \'Gaussian_prior\', \'Laplace_prior\', \'Uniform_prior\'. Default: \'Gaussian_prior\'')
parser.add_argument('--mu', type=float, nargs='?', action='store', default=0.0,
                    help='Mean parameter for prior distribution. Default: 0.0')
parser.add_argument('--sigma', type=float, nargs='?', action='store', default=10.0,
                    help='Standard deviation parameter for prior distribution. Default: 10.0')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='Learning rate for optimizer. Default: 1e-3')
args = parser.parse_args()

#===============================================================

device = torch.device("cuda")

# take the number of epoch as an argument from user, with default value of 10

def save_plot_training_results_with_stats(losses, accuracies, accuracy_epochs, weight_stats, bias_stats, experiment_serial):
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
    plt.savefig(f'results_eurosat/bayesian_cnn_training_results_{experiment_serial}.png')

def predict_data(model, loader_of_interest, num_samples=10):
    model.eval()
    guide.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader_of_interest, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            logits_mc = torch.zeros(num_samples, images.size(0), model.fc1.out_features).to(device)

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

from utils.function import load_data, train_svi_with_stats, plot_training_results_with_stats, send_telegram_message
from utils.model import BayesianCNNSingleFCCustomAdaptive, LaplaceBayesianCNNSingleFCCustomAdaptive, BaseBayesianCNNSingleFCCustom

if __name__ == "__main__":
    # Load data
    experiment_serial = time.strftime("%Y%m%d-%H%M%S")
    
    # Define model
    num_classes = 10
    
    if args.prior == 'Gaussian_prior':
        if args.variant == 'Base':
            bayesian_model = BayesianCNNSingleFCCustom(num_classes=num_classes,
                                            mu=args.mu,
                                            sigma=args.sigma,
                                            device=device).to(device)
        elif args.variant == 'Adaptive':
            bayesian_model = BayesianCNNSingleFCCustomAdaptive(num_classes=num_classes,
                                            mu=args.mu,
                                            sigma=args.sigma,
                                            device=device).to(device)
    elif args.prior == 'Laplace_prior':
        if args.variamt == 'Base':
            bayesian_model = LaplaceBayesianCNNSingleFCCustom(num_classes=num_classes,
                                                mu=args.mu,
                                                b=args.sigma,
                                                device=device).to(device)
        elif args.variant == 'Adaptive':
            bayesian_model = LaplaceBayesianCNNSingleFCCustomAdaptive(num_classes=num_classes,
                                                mu=args.mu,
                                                b=args.sigma,
                                                device=device).to(device)

    #if args.prior == 'Gaussian_prior':
    #    bayesian_model = BaseBayesianCNNSingleFCCustom(num_classes=num_classes,
    #                                        prior_choice="gaussian",
    #                                        mu=args.mu,
    #                                        sigma=args.sigma,
    #                                        device=device).to(device)
    #elif args.prior == 'Laplace_prior':
    #    bayesian_model = BaseBayesianCNNSingleFCCustom(num_classes=num_classes,
    #                                        prior_choice="laplace",
    #                                        mu=args.mu,
    #                                        sigma=args.sigma,
    #                                        device=device).to(device)
    #elif args.prior == 'Uniform_prior':
    #    bayesian_model = BaseBayesianCNNSingleFCCustom(num_classes=num_classes,
    #                                        prior_choice="uniform",
    #                                        mu=args.mu,
    #                                        sigma=args.sigma,
    #                                        device=device).to(device)


    guide = AutoDiagonalNormal(bayesian_model)

    optimizer = Adam({"lr": args.lr})  # Increased from 1e-4 to 1e-3
    svi = pyro.infer.SVI(model=bayesian_model,
                        guide=guide,
                        optim=optimizer,
                        loss=pyro.infer.Trace_ELBO())
    
    pyro.clear_param_store()
    bayesian_model.to(device)
    guide.to(device)

    train_loader, test_loader = load_data()

    losses, accuracies, accuracy_epochs, weight_stats, bias_stats = train_svi_with_stats(
    bayesian_model, guide, svi, train_loader, num_epochs=args.epochs)

    # Plot all results including weight and bias statistics
    save_plot_training_results_with_stats(losses, accuracies, accuracy_epochs, weight_stats, bias_stats, experiment_serial)

    # save the model
    model_path = f'results_eurosat/bayesian_cnn_model_{experiment_serial}.pth'
    torch.save(bayesian_model.state_dict(), model_path)

    # save the guide
    guide_path = f'results_eurosat/bayesian_cnn_guide_{experiment_serial}.pth'
    torch.save(guide.state_dict(), guide_path)

    # save the pyro parameter store
    pyro_param_store_path = f'results_eurosat/pyro_param_store_{experiment_serial}.pkl'
    pyro.get_param_store().save(pyro_param_store_path)

    # predict on test data
    all_labels, all_predictions = predict_data(bayesian_model, test_loader, num_samples=10)
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Accuracy from confusion matrix: {accuracy * 100:.6f}%")

    # save the image index, labels, and predictions in a dataframe
    
    results_df = pd.DataFrame({
        'Image Index': range(len(all_labels)),
        'True Label': all_labels,
        'Predicted Label': all_predictions,
        'Experiment Serial': experiment_serial
    })
    results_df.to_csv(f'results_eurosat/bayesian_cnn_results_{experiment_serial}.csv', index=False)

    # save the training loss and training accuracy per epoch in a dataframe
    training_stats_df = pd.DataFrame({
        'Epoch': accuracy_epochs,
        #'Loss': losses,
        'Training Accuracy': accuracies
    })
    training_stats_df.to_csv(f'results_eurosat/bayesian_results_stats{experiment_serial}.csv', index=False)

    # save the training configuration in a json file
    config = {
        'experiment_serial': experiment_serial,
        'num_epochs': args.epochs,
        'prior': args.prior,
        'mu': args.mu,
        'sigma': args.sigma,
        'learning_rate': args.lr
    }

    with open(f'results_eurosat/bayesian_cnn_config_{experiment_serial}.json', 'w') as f:
        json.dump(config, f, indent=4)

    send_telegram_message(
        title="Training Completed for Experiment " + experiment_serial,
        message=f"Training configuration: \n{config}\n"
                f"Train acuracies: {accuracies}\n"
                f"Best training accuracy: {max(accuracies) * 100:.6f}%\n"
                f"At epoch: {accuracy_epochs[np.argmax(accuracies)]}\n"
                f"Confusion Matrix Accuracy: {accuracy * 100:.6f}%\n"
                #f"Results plot saved as: results_eurosat/bayesian_cnn_training_results_{experiment_serial}.png"
    )