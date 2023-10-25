import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(df, nrows=3, nbins=30):
    ncols = np.ceil(len(df.columns) / nrows).astype(int)
    #print(nrows, ncols, len(df.columns), len(df.columns)/nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3.0, nrows*2.75))
    #plt.subplots_adjust(wspace=0.5, hspace=1)  # Adjust the spacing

    # Iterate through columns and add histograms and vertical lines
    for i, col in enumerate(df.columns):
        row, col_idx = divmod(i, ncols)
        if nrows == 1:
            ax = axes[i]
        else:
            ax = axes[row, col_idx]
        
        ax2 = ax.twinx()
        if col != 'Potability':
            df[col].hist(ax=ax, bins=nbins)
            df[col].hist(ax=ax2, bins=nbins, cumulative=True, density=1, histtype='step', color='tab:orange')
            df_n_mean = df[col].mean()
            df_n_std = df[col].std()
            ax.axvline(x=df_n_mean-3*df_n_std, color='darkblue', linestyle='--') 
            ax.axvline(x=df_n_mean+3*df_n_std, color='darkblue', linestyle='--') 
            ax.axvline(x=df_n_mean, color='darkblue', linestyle='--') 
            ax.axvline(x=df[col].median(), color='red', linestyle=':') 
            ax2.axhline(y=0.5, color='black', linestyle=':') 
        else:
            df[col].hist(ax=ax, bins=2)
            df[col].hist(ax=ax2, bins=2, cumulative=True, density=1, histtype='step', color='tab:orange')
        ax.set_title(col)

        ax.grid(False)
        ax2.grid(False)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, nrows=1, figsize=(7, 3)):
    ncols = np.ceil(len(df.columns) / nrows).astype(int)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.4, nrows*4))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the spacing

    # Iterate through columns and add histograms and vertical lines
    for i, col in enumerate(df.columns):
        row, col_idx = divmod(i, ncols)
        if nrows == 1:
            ax = axes[i]
        else:
            ax = axes[row, col_idx]
        df[[col]].boxplot(
            ax=ax, 
            showmeans=True,
            #meanline=True,
            medianprops=dict(color='r'),
            meanprops=dict(
                markeredgecolor='blue',
                #markerfacecolor='black' ,
                marker='x',
                # alpha=.5,
            ),
        )
    plt.tight_layout()
    plt.show()

def plot_boxplots_normalize_df(df, figsize=(7, 3), rotation=45):
    normalized_df = (df - df.mean()) / df.std()
    # Create subplots with different y-axes for each column
    normalized_df.boxplot(
            figsize=figsize,
            showmeans=True,
            #meanline=True,
            medianprops=dict(color='r'),
            meanprops=dict(
                markeredgecolor='blue',
                #markerfacecolor='black' ,
                marker='x',
                # alpha=.5,
            ),
        )
    plt.axhline(y=0, color='blue', linestyle=':')
    plt.xticks(rotation=45)
    plt.show()