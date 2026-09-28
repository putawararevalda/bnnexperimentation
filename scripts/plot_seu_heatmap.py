import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_dataset(pattern):
    dfs = []
    for f in glob.glob(pattern):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    # Only keep b=1.0 configs for apples-to-apples comparison
    df = df[df['prior_b'] == 1.0].copy()
    df['aad'] = df['accuracy_change'].abs()
    
    # Calculate ARIn
    g = df.groupby(['location_layer', 'location_module', 'bit_index'])
    aad_mean = g['aad'].mean()
    smd_mean = g['softmax_difference'].mean()
    arin = ((aad_mean**2 + smd_mean**2) / 2)**0.5
    
    heatmap_df = arin.unstack()
    # Sort y-axis
    heatmap_df.index = [f"{layer}.{module}" for layer, module in heatmap_df.index]
    order = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias']
    heatmap_df = heatmap_df.reindex(order)
    
    return heatmap_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='images/seu_heatmap.pdf')
    args = parser.parse_args()
    
    print("Loading ShipsNet data...")
    shipsnet_df = load_dataset('results/shipsnet/seu/fold*/*/*.csv')
    
    print("Loading EuroSAT data...")
    eurosat_df = load_dataset('results/eurosat/seu_clean/*/*.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    vmax = max(shipsnet_df.max().max(), eurosat_df.max().max())
    
    sns.heatmap(shipsnet_df, ax=axes[0], cmap='Reds', vmin=0, vmax=vmax,
                annot=True, fmt=".3f", cbar=False, annot_kws={"size": 9})
    axes[0].set_title('ShipsNet (ARIn)')
    axes[0].set_xlabel('Bit Index')
    axes[0].set_ylabel('Injection Site')
    
    sns.heatmap(eurosat_df, ax=axes[1], cmap='Reds', vmin=0, vmax=vmax,
                annot=True, fmt=".3f", cbar=True, annot_kws={"size": 9})
    axes[1].set_title('EuroSAT (ARIn)')
    axes[1].set_xlabel('Bit Index')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.savefig(args.out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {args.out}")


if __name__ == '__main__':
    main()
