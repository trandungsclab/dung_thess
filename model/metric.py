import torch
from sklearn import metrics
from csv import writer
import pandas as pd
import pathlib

def append_list_as_row( list_of_elem):
        file_name = "result_ananysis/non_speaker/train_primary.csv"
        f = pathlib.Path(file_name)
        if not f.exists():
            col = ["Actual", "Prediction"]
            data= pd.DataFrame([], columns=col)
            data.to_csv(file_name, index=None)    
        with open(file_name, 'a+', newline='') as write_obj:
             csv_writer = writer(write_obj)
             csv_writer.writerow(list_of_elem)

def mse(output, target):
    with torch.no_grad():
        return metrics.mean_squared_error(list(target.cpu().numpy()), list(output.cpu().numpy()))

def macro_f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        return metrics.f1_score(list(target.cpu().numpy()), list(pred.cpu().numpy()), average='macro')

def weighted_f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        return metrics.f1_score(list(target.cpu().numpy()), list(pred.cpu().numpy()), average='weighted')

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        
        for i in range(len(target)):
            # import pdb; pdb.set_trace()
            append_list_as_row([target[i].item(),pred[i].item()])
            
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
