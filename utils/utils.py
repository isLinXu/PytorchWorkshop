import os
import numpy as np
import pandas as pd
import yaml
import random
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def load_data_info(yaml_file):
    with open(yaml_file, "r") as stream:
        yaml_data = yaml.load(stream, Loader=yaml.FullLoader)
    
    train_file = yaml_data["train_file"]
    train_csv = yaml_data["train_csv"]
    
    valid_file = yaml_data["valid_file"]
    valid_csv = yaml_data["valid_csv"]
    
    classes_len = yaml_data["nc"]
    classes_names = yaml_data["names"]

    train_df = pd.read_csv(train_csv)
    
    if (valid_file == "None") and (valid_csv == "None"):
        train_set, valid_set = train_test_split(train_df, stratify=train_df['label'], train_size=0.8, random_state=42)
        valid_file = train_file
        
    elif valid_file != "None" and valid_csv != "None":
        valid_df = pd.read_csv(valid_csv)
        train_set = train_df
        valid_set = valid_df
        
    else:
        raise AssertionError('!!! Please check audio.yaml !!!')
    
    return (train_file, valid_file), (classes_len, classes_names), (train_set, valid_set)   
    
def init_distributed_mode(local_rank, cuda_num):
    if 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
        
        device = torch.device("cuda", local_rank)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
    elif torch.cuda.is_available():
        device = torch.device("cuda", cuda_num)
        rank = 0
        world_size = 1
        
    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        
    return device, rank, world_size
    
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def visualization(num_epochs, train, valid, title, model_saved_path):
    plt.plot(range(num_epochs), train, 'b-', label=f'Training_{title}')
    plt.plot(range(num_epochs), valid, 'g-', label=f'validation{title}')
    plt.title(f'Training & Validation {title}')
    plt.xlabel('Number of epochs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(os.path.join(model_saved_path, f"{title}.jpg"))
    plt.close()
    
def confusion_matrix_plot(y_true, y_pred, classes_names, model_saved_path):
    plt.figure(figsize=(12,8))
    classes_number = [i for i in range(len(classes_names))]
    cf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy(), labels=classes_number)
    cf_matrix = pd.DataFrame(cf_matrix, index = classes_names, columns = classes_names)  
    sns.heatmap(cf_matrix.T, cmap='Blues', annot=True)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.savefig(os.path.join(model_saved_path, "confusion_matrix.jpg"))
    plt.close()