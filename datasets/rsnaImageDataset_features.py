
import numpy as np
import pickle
import torch
import os

from torch.utils.data import Dataset, DataLoader
from csv_dataset import CsvFunctions
from torch.utils.data import Dataset
from torchvision import transforms


# tuple[transforms] == typerror
def generate_transforms(input_size, is_train=True):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([input_size, input_size]),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([input_size, input_size]),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


SERIES_IDX = 9
POSITION_Z_IDX = -1


def group_slices_by_series(data):
    series = {}
    for line in data:
        serie_id = line[SERIES_IDX]
        if serie_id in series:
            series[serie_id].append(line)
        else:
            series[serie_id] = [line]
    return series


def sort_series_by_z_position(series):
    for key, val in series.items():
        series[key] = sorted(val, key=lambda x: float(x[POSITION_Z_IDX]))
    return series


class RsnaDatasetImage(Dataset): # CnnFeaturesDataset
    def __init__(self, csv_file: str, features_pickles_dir: str):
        self.features_pickles_dir = features_pickles_dir
        data = CsvFunctions.load(csv_file)
        series = group_slices_by_series(data)
        series = sort_series_by_z_position(series)

        '''
        series = 
        [ 
            [ID_123, ID_234, ID_456, ...],
            [ID_123, ID_234, ID_456, ...],
            [ID_123, ID_234, ID_456, ...],
            ...
        ]
        '''
        self.series = list(series.items())

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        '''
            o que deve ser retornado pela funcao:
                entradas da rede: shape: [n_slices, tamanho features]
                    [
                        [0.9, 0.54, 0.3], <- features do slice ID_123
                        [0.2, 0.11, 0.02], <- features do slice ID_234
                        ...
                    ]
                saídas esperadas da rede: label (tem ou não hemorragia)
            return features_slices_series_index, labels_slices
        '''
        features_pickles_dir = self.features_pickles_dir
        serie_ID, list_slices = self.series[index]
        serie_features = []

        serie_label = []
        for slice in list_slices:
            slice_ID = slice[0]
            slice_path = os.path.join(features_pickles_dir, slice_ID + ".pkl")
            
            file_features = open(slice_path, "rb")
            img_feature = pickle.load(file_features)
            file_features.close()
            
            slice_label = slice[1:7] # any == is there hemorrhage

            serie_features.append(img_feature)
            
            if serie_label == []:
                serie_label = slice_label # np.array(slice_label, dtype=float)
            else:
                for pos in range(len(slice_label)):
                    if slice_label[pos] == '1':
                        serie_label[pos] = '1'
        
        # if np.array(serie_features).shape[0] > 60:
        #     print(f'A imagem {slice_ID} tem {np.array(serie_features).shape[0]} slices.')
        
        while np.array(serie_features).shape[0] < 60: # 60 is the max value of slices in a serie
            blank_slice = np.zeros(
                np.array(serie_features).shape[1] # size of feature maps
                )
            serie_features.append(blank_slice)

        # if np.array(serie_features).shape[0] != 60:
        #     print(f'A imagem {slice_ID} tem um número de slices diferente de 60 ({np.array(serie_features).shape[0]} slices).')
        return torch.tensor(np.array(serie_features), dtype=torch.float), torch.tensor(np.array(serie_label, dtype=float), dtype=torch.float)


def generate_dataset(dataset_path,
                     dataset_lines):

    train_dataset = RsnaDatasetImage(dataset_path,
                                     dataset_lines)

    return train_dataset


def generate_loader(dataset,
                    batch_size=64,
                    val_dataset=None,
                    val_batch_size=64,
                    workers=2,
                    is_train=True):

    if is_train == False:
        test_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True)
        
        return test_loader
    
    else:
        train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=workers,
                                pin_memory=True)

        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True)

        return train_loader, val_loader
