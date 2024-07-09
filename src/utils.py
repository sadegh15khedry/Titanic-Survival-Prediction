import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf

def save_confution_matrix(cm, file_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(file_path)
    
    
def save_report(report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
        
def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def save_model(model, path):
    model.save(path)
    

def load_saved_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    


def set_pandas_display_options():
    pd.set_option('display.max_columns', 20)
    # Increase the maximum width of the display
    pd.set_option('display.width', 1000)
    
    
def get_error(y_train, y_pred_train):
    mse_train = mean_squared_error(y_train, y_pred_train)
    print(f"Training MSE: {mse_train}")
    
    
def load_data(file_name):
    return pd.read_csv(file_name)

def test_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))