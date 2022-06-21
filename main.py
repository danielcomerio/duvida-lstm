import torch.nn as nn
import numpy as np
import datetime
import torch
import time
import os

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from typing import Any, List, Tuple, Callable
from tqdm import tqdm

from settingsFilipe import (BATCH_SIZE, EPOCHS, LEARNING_RATE, OPTIMIZATION_FUNCTION,
                      DATASET_TEXT_PATH, FEATURES_PATH, TRAINED_MODELS_PATH)
from models import initialize_pretrained_model
from datasets import rsnaImageDataset_features


class LstmModel(torch.nn.Module):
  def __init__(self, feature_size, num_classes):
    super().__init__()
    self.lstm = torch.nn.LSTM(input_size=feature_size, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
    self.fc = torch.nn.Linear(512, num_classes) # 512 pq eh bidirecional
  
  def forward(self, x):
    _, (hn, _) = self.lstm(x)
    hn = hn.view(x.cpu().detach().numpy().shape[0], FEATURE_SIZE) # muda o shape para 32x512             batch 64       (2 é o número de series atual utilizado)
    output = self.fc(hn)
    return output


def evaluate_features(model: Any,
                      model_name: Any,
                      device: Any,
                      test_loader: Any,
                     ) -> Tuple[List[int], List[int]]:
    
    path = os.path.join("model_info", "test_metrics")
    if not os.path.isdir(path):
        os.makedirs(path)
    
    test_metrics_txt = open(os.path.join(path, f"lstm_{model_name}.txt"), 'w')
    
    dict_hemo = {"epidural":        {"predicted": [], "labels": []},
                "intraparenchymal": {"predicted": [], "labels": []},
                "intraventricular": {"predicted": [], "labels": []},
                "subarachnoid":     {"predicted": [], "labels": []},
                "subdural":         {"predicted": [], "labels": []},
                "any":              {"predicted": [], "labels": []}
                }
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(test_loader)):
            features = features.to(device)
            logits = model(features)
            logits = logits.to("cpu")

            for i in range(len(logits)):
                currently_label = labels[i]

                pred = [(1 if elem >= 0.5 else 0) for elem in logits[i]]

                for pos, hemo in enumerate(dict_hemo):
                    dict_hemo[hemo]["predicted"].append(pred[pos])
                    dict_hemo[hemo]["labels"].append(currently_label[pos])
    
    
    print()
    print("Test confusion matrix")
    print()
    test_metrics_txt.write("Test confusion matrix\n\n")
    for pos, hemo in enumerate(dict_hemo):
        val_preds = dict_hemo[hemo]["predicted"]
        val_trues = dict_hemo[hemo]["labels"]
        
        print(f"{hemo} metrics:")
        print(classification_report(val_trues, val_preds))
        print(f"{hemo} confusion matrix:")
        print(confusion_matrix(val_trues, val_preds))
        print()
        print()

        test_metrics_txt.write(f"{hemo} metrics:\n")
        test_metrics_txt.write(classification_report(val_trues, val_preds))
        test_metrics_txt.write('\n')
        
        test_metrics_txt.write(f"{hemo} confusion matrix:\n")
        [test_metrics_txt.write(
            str(row) + '\n') for row in confusion_matrix(val_trues, val_preds)]
        test_metrics_txt.write('\n\n\n\n\n')
        
    test_metrics_txt.close()
    

def train_features(model: Callable,
                model_name: str,
                device: Any,
                train_loader: Any,
                val_loader: Any) -> Tuple[List[List], List[List]]:

    folders_to_create = ["models_graph", "epoch_metrics"]
    for fold in folders_to_create:
        path = os.path.join("model_info", fold)
        if not os.path.isdir(path):
            os.makedirs(path)

    if not os.path.isdir(TRAINED_MODELS_PATH):
        os.makedirs(TRAINED_MODELS_PATH)

    trains_info_txt = open(os.path.join(
        "model_info", "trainings_duration.txt"), 'w')
    epoch_train_metrics_txt = open(os.path.join(
        "model_info", "epoch_metrics", f"lstm_{model_name}.txt"), 'w')
    

    #loss_fn = LOSS_FUNCTION(reduction='sum')
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = OPTIMIZATION_FUNCTION(model.parameters(), lr=LEARNING_RATE)

    train_losses, train_hemo_accuracies = [], []
    val_losses, val_hemo_accuracies = [], []

    start_time = time.time()
    best_val_f1 = -1
    best_epoch = -1
    for epoch in range(EPOCHS):
        epoch_time = time.time()
        train_loss = 0
        train_hemo_hits = 0
        

        dict_hemo = {"epidural":         {"predicted": [], "labels": []},
                     "intraparenchymal": {"predicted": [], "labels": []},
                     "intraventricular": {"predicted": [], "labels": []},
                     "subarachnoid":     {"predicted": [], "labels": []},
                     "subdural":         {"predicted": [], "labels": []},
                     "any":              {"predicted": [], "labels": []}
                    }

        num_features_train = 0
        model.train()
        for batch_idx, (features, labels) in tqdm(enumerate(train_loader)):
            num_features_train += len(labels)
            features = features.to(device)
            #label = label.to(device)
            optimizer.zero_grad()  # reinitialize gradients
            logits = model(features)

            loss = 0
            for i in range(len(logits)):
                currently_label = labels[i]
                loss += loss_fn(logits[i], torch.Tensor(currently_label).to(device))

                pred = [(1 if elem >= 0.5 else 0) for elem in logits[i]]
                
                train_hemo_hits += 1 if pred[5] == currently_label[5] else 0 # any
                
            loss.backward()  # compute gradients in a backward pass
            optimizer.step()  # update weights
            loss = loss.item()
            train_loss += loss
            
        train_loss /= num_features_train
        train_losses.append(train_loss)
        train_hemo_acc = train_hemo_hits / num_features_train
        train_hemo_accuracies.append(train_hemo_acc)

        val_loss = 0
        val_hemo_hits = 0
        val_frac_hits = 0
        val_preds = []
        val_trues = []

        num_features_val = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (features, labels) in tqdm(enumerate(val_loader)):
                num_features_val += len(labels)
                features = features.to(device)
                #label = label.to(device)
                logits = model(features)
                
                loss = 0
                for i in range(len(logits)):
                    currently_label = labels[i]
                    currently_label_int = [int(elem) for elem in currently_label]
                    loss += loss_fn(logits[i], torch.Tensor(currently_label).to(device))

                    pred = [(1 if elem >= 0.5 else 0) for elem in logits[i]]

                    for pos, hemo in enumerate(dict_hemo):
                        dict_hemo[hemo]["predicted"].append(pred[pos])
                        dict_hemo[hemo]["labels"].append(currently_label_int[pos])

                    val_hemo_hits += 1 if int(pred[5]) == int(currently_label_int[5]) else 0 # any
                
                val_loss += loss

        total_epoch_time = time.time() - epoch_time
        print(f"\nDuração da época {epoch}: {str(datetime.timedelta(seconds=total_epoch_time))}\n\n")

        val_loss /= num_features_val
        val_losses.append(val_loss)
        val_hemo_acc = val_hemo_hits / num_features_val
        val_hemo_accuracies.append(val_hemo_acc)

        val_preds_any = dict_hemo["any"]["predicted"]
        val_trues_any = dict_hemo["any"]["labels"]
        val_any_f1 = f1_score(val_trues_any, val_preds_any)

        if val_any_f1 > best_val_f1:
            torch.save(model, os.path.join(
                TRAINED_MODELS_PATH, f"lstm_{model_name}.pt"))
            print("Saved best f1 model (any class).")
            best_val_f1 = val_any_f1
            best_epoch = epoch

        val_hemo_precision = precision_score(val_trues_any, val_preds_any)
        val_hemo_recall = recall_score(val_trues_any, val_preds_any)

        print(f"epoch: {epoch} | loss: {train_loss:.3f}, hemo_acc: {train_hemo_acc:.3f} || val_loss: {val_loss:.3f}, val_hemo_acc: {val_hemo_acc:.3f}, val_hemo_precision: {val_hemo_precision:.3f}, val_hemo_recall: {val_hemo_recall:.3f}, val_any_f1: {val_any_f1:.3f}")

        epoch_train_metrics_txt.write(
            f"epoch: {epoch} | loss: {train_loss:.3f}, hemo_acc: {train_hemo_acc:.3f} || val_loss: {val_loss:.3f}, val_hemo_acc: {val_hemo_acc:.3f}, val_hemo_precision: {val_hemo_precision:.3f}, val_hemo_recall: {val_hemo_recall:.3f}, val_any_f1: {val_any_f1:.3f}")

        print()
        print("Validation confusion matrix")
        print()
        epoch_train_metrics_txt.write("Validation confusion matrix\n\n")
        for pos, hemo in enumerate(dict_hemo):
            val_preds = dict_hemo[hemo]["predicted"]
            val_trues = dict_hemo[hemo]["labels"]
            
            print(f"{hemo} confusion matrix:")
            print(confusion_matrix(val_trues, val_preds))
            print()
            print()

            epoch_train_metrics_txt.write(f"{hemo} confusion matrix:\n")
            [epoch_train_metrics_txt.write(
                str(row) + '\n') for row in confusion_matrix(val_trues, val_preds)]
            epoch_train_metrics_txt.write('\n')
            epoch_train_metrics_txt.write('\n')
        
        epoch_train_metrics_txt.write('\n\n\n\n')
        
        
    total_time_train = time.time() - start_time
    trains_info_txt.write(
        f'Tempo total de treino da rede "{model_name}" -> {str(datetime.timedelta(seconds=total_time_train))}\n\n\n')

    print(f'Tempo total de treino da rede "{model_name}" -> {str(datetime.timedelta(seconds=total_time_train))}\n\n')
    print("----------------\n\n\n\n")
    
    epoch_train_metrics_txt.write(f"\n\n\n\nBest Epoch: {best_epoch}\n\n")
    epoch_train_metrics_txt.close()

FEATURE_SIZE = -1
def main():
    model_names = ["densenet121", "densenet169"]
    
    for model_name in model_names:
        print(model_name)
        _, input_size, feature_size = initialize_pretrained_model(model_name, num_classes=6)
        FEATURE_SIZE = feature_size
        
        model = LstmModel(feature_size, num_classes=6)

        train_dataset = rsnaImageDataset_features.generate_dataset(
                    DATASET_TEXT_PATH.joinpath("train.csv"),
                    FEATURES_PATH.joinpath(model_name)
                    )
        
        val_dataset = rsnaImageDataset_features.generate_dataset(
                    DATASET_TEXT_PATH.joinpath("val.csv"),
                    FEATURES_PATH.joinpath(model_name)
                    )
        
        test_dataset = rsnaImageDataset_features.generate_dataset(
                    DATASET_TEXT_PATH.joinpath("test.csv"),
                    FEATURES_PATH.joinpath(model_name)
                    )
        
        train_loader, val_loader = rsnaImageDataset_features.generate_loader(train_dataset, BATCH_SIZE, val_dataset, BATCH_SIZE)
        test_loader = rsnaImageDataset_features.generate_loader(test_dataset, batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE, is_train=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        train_features(model, model_name, device, train_loader, val_loader)
        
        evaluate_features(model, model_name, device, test_loader)
        
        print("----------------\n\n\n\n\n\n\n\n\n")


if __name__ == "__main__":
    main()
