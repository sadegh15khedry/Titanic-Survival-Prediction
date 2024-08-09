import matplotlib.pyplot as plt
import seaborn as sns


def visualize_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('../results/correlation_matrix.png')
    plt.show()
    
    
    
def get_columns_values_counts(df):
    for column in df.columns:
        print(column)
        print(df[column].value_counts() , '\n')