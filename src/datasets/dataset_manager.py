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
import ast

def generate_random_template():
    # 定义不同的句子模板
    templates = [
        "The reaction type is ",
        "Given that the reaction type is ",
        "Considering the reaction type is ",
        "What would be the outcome if the reaction type is "
    ]
    
    # 从句子模板中随机选择一个
    selected_template = random.choice(templates)
    
    return selected_template

def merge_graphs(data_list):
    # 初始化合并后的图的节点特征和边索引
    x = torch.cat([data.x for data in data_list], dim=0)
    
    edge_index_list = []
    edge_attr_list = []  # 如果有边特征
    node_offset = 0  # 节点索引的偏移量
    for data in data_list:
        # 调整边索引并添加到列表
        edge_index_list.append(data.edge_index + node_offset)
        # 如果有边特征，也合并它们
        if data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr)
        # 更新节点索引的偏移量
        node_offset += data.num_nodes
    
    # 合并边索引
    edge_index = torch.cat(edge_index_list, dim=1)
    # 合并边特征（如果有）
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None

    # 创建新的 Data 对象
    merged_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return merged_data

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        #print(mol)
        if mol is not None:
            return True
        else:
            return False
    except:
        return False

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

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
    def __init__(self, split=None, tokenizer_org = None, args = None, transform=None):
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
        self.moe = args.moe
        self.scaffold_split = args.scaffold_split 
        self.tokenizer_org = tokenizer_org

        graph2d_featurizer_config = { 
            'name' : 'ogb'
        }
        self.task = args.task
        self.input_modal = args.input_modal
        self.graph2d_featurizer = GraphFeaturizer(graph2d_featurizer_config)

        super(MolDataset, self).__init__(config=None)
                                               #for bond in mol.GetBonds())))

    def _load_data(self):
        self.instruction = []
        self.input = []
        self.output = []
        self.task = []
        self.graph2d = []
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
            self.instruction.append(row["instruction"])
            self.input.append(row["input"])
            self.output.append(row["output"])
            self.task.append(row['metadata']["task"])
            self.id.append(row["id"])
            if(self.input_modal == 'graph'):
                compounds_smiles = re.split(r'\.|\>\>', row["input"])
                compounds_smiles = [smi for smi in compounds_smiles if smi]
                graph_features = [self.graph2d_featurizer(smi) for smi in compounds_smiles]
                merge_graph = merge_graphs(graph_features)
                self.graph2d.append(merge_graph)
            try:
                self.types.append(row["reaction_label_5"])
            except:
                self.types.append(-1)
                #self.types.append(row["reaction_label_5"])
            
    def return_df(self):  
        return self.data
        
    def __getitem__(self, i):
        mol_rc_data = {}
        #input_modal_data
        if(self.reaction_type == True):
            #instruction = f"{self.instruction[i]}& {generate_random_template()}{self.types[i]}."
            #print(instruction)
            mol_rc_data['instruction'] = self.tokenizer_org(
                self.instruction[i],
                return_tensors="pt"
            )
        else:
            mol_rc_data['instruction'] = self.tokenizer_org(
                self.instruction[i],
                return_tensors="pt"
            )
        
        mol_rc_data['type']=self.types[i]
        
        mol_rc_data['output'] = self.tokenizer_org(
                self.output[i],
                return_tensors="pt"
            )
        
        mol_rc_data['id'] = self.id[i]
        
        mol_rc_data['input'] = self.tokenizer_org(
            self.input[i],
            return_tensors="pt"
        )
            
        if(self.split in ['valid', 'test']):
            mol_rc_data['task'] = self.task[i]
            mol_rc_data['truth'] = self.output[i]
            
        return mol_rc_data
