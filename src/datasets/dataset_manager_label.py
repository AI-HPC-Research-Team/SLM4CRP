# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from feature.graph_featurizer import GraphFeaturizer

import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import re
import numpy as np
import random
from rdkit.Chem.rdchem import HybridizationType, BondType
from torch_geometric.data import Data


class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self._load_data()
        
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return self.data_shape

    
class MolDataset(BaseDataset):
    """
    This function processes the dataset for machine learning tasks.
    
    Parameters:
        data_path (str): The path to the source data file. Supported formats include pkl, csv, and txt. The data will be read using pandas.
        
        config (dict): A dictionary containing various settings and configurations for data processing.
        
        split (str): Specifies the type of dataset to be returned. Options are 'train', 'valid', and 'test'.
        
        tokenizer_org (Tokenizer, optional): A tokenizer for processing the source data. If None, a default tokenizer will be used.
        
        tokenizer_label (Tokenizer, optional): A tokenizer for processing the label data. If None, the tokenizer_org will be used.
        
        task (dict, optional): Task in this project, Specifies the modality of the input and output data.
        
        transform (callable, optional): A function/transform to apply to image data for normalization or data augmentation.
    
    Returns:
        processed_data (Dataset): A processed dataset ready for training or evaluation.
    """
    def __init__(self, df = None, split=None, tokenizer_org = None, args = None, transform=None):
        if(df is not None):
            self.data = df
        else:
            data_path = args.dataset
            if data_path.endswith('.pkl'):
                self.data = pd.read_pickle(data_path)
            elif data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            elif data_path.endswith('.txt'):
                self.data = pd.read_table(data_path)
            elif data_path.endswith('.json'):
                self.data = pd.read_json(data_path)
            else:
                raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.split = split
        self.reaction_type = args.reaction_type
        self.tokenizer_org = tokenizer_org
        #self.task = args.task
        self.args = args

        super(MolDataset, self).__init__(config=None)
                                               #for bond in mol.GetBonds())))

    def _load_data(self):
        self.instructions = []
        self.input = []
        self.output = []
        self.tasks = []
        self.id = []
        self.types = []
        if(self.split == 'all'):
            filtered_data = self.data
        else:
            filtered_data = self.data[self.data['metadata'].apply(lambda x: x.get('split') if isinstance(x, dict) else None) == self.split]

        # 更新data_shape为筛选后的数据量
        self.data_shape = len(filtered_data)
        self.data = filtered_data
        # 使用tqdm包装filtered_data.iterrows()来显示进度条
        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.input.append(row["input"])
            self.output.append(row["output"])
            self.tasks.append(row['metadata']["task"])
            self.id.append(row["id"])
            if(self.reaction_type == True):
                self.types.append(row["reaction_label"])
                prompt = f"This is the {self.args.task} Reaction Prediction Task, where the goal is to determine the type of chemical reaction based on the given compounds, categorized as 0 through {self.args.cluster_number}."
                self.instructions.append(prompt)

            
    def return_df(self):  
        return self.data
        
    def __getitem__(self, i):
        mol_rc_data = {}
            
        if(self.reaction_type == True):
            mol_rc_data['type']=self.types[i]
            
            mol_rc_data['label'] = self.tokenizer_org(
                    str(self.types[i]),
                    return_tensors="pt"
                )
            
            mol_rc_data['prompt'] = self.tokenizer_org(
                    self.instructions[i],
                    return_tensors="pt"
                )
            
        mol_rc_data['output'] = self.tokenizer_org(
                self.output[i],
                return_tensors="pt"
            )

        mol_rc_data['id'] = self.id[i]
        
        mol_rc_data['input'] = self.tokenizer_org(
            self.input[i],
            return_tensors="pt"
        )
            
            
        return mol_rc_data
