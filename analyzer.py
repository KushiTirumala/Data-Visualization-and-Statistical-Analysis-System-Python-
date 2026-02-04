import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from scipy import stats

def load_data(file):
    return pd.read_csv(file)

def basic_stats(df):
    return {
        'mean': df.mean(numeric_only=True),
        'median': df.median(numeric_only=True),
        'std': df.std(numeric_only=True),
        'corr': df.corr(numeric_only=True)
    }

def plot_visuals(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    df.hist(ax=axes[0,0])
    axes[0,0].set_title('Distribution')
    
    # Scatter
    if len(df.columns) >= 2:
        axes[0,1].scatter(df.iloc[:,0], df.iloc[:,1])
        axes[0,1].set_title('Scatter Plot')
    
    # Boxplot
    df.boxplot(ax=axes[1,0])
    axes[1,0].set_title('Box Plot')
    
    # Time series if date col
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date').plot(x='date', y=df.columns[1], ax=axes[1,1])
        axes[1,1].set_title('Trend')
    
    plt.tight_layout()
    plt.savefig('visuals.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Data Viz & Stats with Pandas/Matplotlib")
    parser.add_argument("file", help="CSV file")
    args = parser.parse_args()
    
    df = load_data(args.file)
    print("Dataset shape:", df.shape)
    print("\nBasic Stats:")
    print(basic_stats(df))
    
    if len(df.select_dtypes(include='number').columns) > 1:
        print("\nCorrelation Matrix:")
        print(df.corr(numeric_only=True))
    
    plot_visuals(df)
    print("Visuals saved as visuals.png")

if __name__ == "__main__":
    main()
