# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models import SUPPORTED_Model, SUPPORTED_CKPT, PromptManager
from models.momu import MoMu
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
from torch.nn.functional import one_hot


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[128, 64], num_classes=10, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
 
        self.layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        self.layers.append(nn.Dropout(dropout_rate)) 
        

        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            self.layers.append(nn.Dropout(dropout_rate))  
        

        self.layers.append(nn.Linear(hidden_layer_sizes[-1], num_classes))
    
    def forward(self, x):

        for i in range(0, len(self.layers) - 1, 2):
            x = F.relu(self.layers[i](x))
            x = self.layers[i+1](x)  

        x = self.layers[-1](x)
        return x


class PromptSelectionModel(nn.Module):
    def __init__(self, config = None):
        super(PromptSelectionModel, self).__init__()
       
        self.config = config
        
    def _select_prompts_based_on_type(self, type_idx, prompt_embeddings):

        if 0 <= type_idx <= 9:
            return prompt_embeddings[0:12]
        elif 10 <= type_idx <= 19:
            return prompt_embeddings[12:24]
        elif 20 <= type_idx <= 29:
            return prompt_embeddings[24:36]
        else:
            raise ValueError("Invalid type index")

    def forward(self, types, input_embeddings, prompt_embeddings):
        total_loss = 0
        for i, type_idx in enumerate(types):
            selected_prompts = self._select_prompts_based_on_type(type_idx, prompt_embeddings)
            input_expanded = input_embeddings[i].unsqueeze(0).unsqueeze(1).expand(-1, selected_prompts.size(0), -1)
            prompt_expanded = selected_prompts.unsqueeze(0)


            similarity = -torch.norm(input_expanded - prompt_expanded, dim=2)
     
            total_loss += -similarity.mean()
   
        avg_loss = total_loss / len(types)
        return avg_loss

    def get_best_prompt_indices(self, types, input_embeddings, prompt_embeddings):
        best_prompt_indices = []
        for i, type_idx in enumerate(types):
            selected_prompts = self._select_prompts_based_on_type(type_idx, prompt_embeddings)
            input_expanded = input_embeddings[i].unsqueeze(0).unsqueeze(1).expand(-1, selected_prompts.size(0), -1)
            prompt_expanded = selected_prompts.unsqueeze(0)

            similarity = -torch.norm(input_expanded - prompt_expanded, dim=2)

            _, best_idx = similarity.max(1)
            best_prompt_indices.append(best_idx.item())
        return best_prompt_indices

class MolModel(nn.Module):
    def __init__(self, config=None):
        super(MolModel, self).__init__()
        self.config = config
        self.config.hidden_size = 768
        self.language_model  = SUPPORTED_Model[self.config.model_name](self.config)
        self.config.gnn_type = 'gin'
        self.fc_2d = nn.Linear(in_features=self.config.hidden_size, out_features=2)
        self.reaction_type = self.config.reaction_type
        self.mode = self.config.mode
        self.PromptManager = PromptManager(self.config)
        self.tokenizer = self.language_model.tokenizer
        self.prompts_tensor = self.process_prompts()
        self.PromptSelectionModel = PromptSelectionModel(self.config)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        if(self.mode == 'data_label'):
            self.classifier = MLPClassifier(input_size=self.config.hidden_size)
            self.criterion = nn.CrossEntropyLoss()
            
        if(self.config.freeze == True):
            for k, v in self.language_model.named_parameters():
                v.requires_grad = False


    def forward(self, mol):
        #h, h_attention_mask = self.encode_h(mol) 
        prompt_selection_loss, h, h_attention_mask = self.encode_h_dy(mol) 
        labels = {}
        #labels["input_ids"] = mol[f"{self.output_modal}_labels"]["input_ids"].masked_fill(~mol[f"{self.output_modal}_labels"]["attention_mask"].bool(), -100)
        labels["input_ids"] = mol["output"]["input_ids"]
        labels["attention_mask"] = mol["output"]["attention_mask"]

        h = BaseModelOutput(
            last_hidden_state= h,
            hidden_states= None,
            attentions= None
        )
        task_loss = self.language_model(
            encoder_outputs = h,
            attention_mask = h_attention_mask,
            decoder_attention_mask = labels["attention_mask"],
            labels = labels["input_ids"]
        )
        
        #alpha = torch.sigmoid(self.alpha)
        #beta = torch.sigmoid(self.beta)
        
        #total_loss =  task_loss +  prompt_selection_loss
        total_loss = task_loss
        return total_loss
    
    def encode_h(self, mol):
        embeddings = self.encode_embeddings(mol)
        #labels = mol[f"{self.output_modal}_labels"]
 
        h_return = torch.cat([embeddings['prompt'], embeddings['text']], dim=1)
        h_attention_mask = torch.cat([mol['instruction']["attention_mask"], mol['input']["attention_mask"]], dim=1)
        
        return h_return, h_attention_mask
    
    def encode_h_label(self, mol):
        embeddings = self.encode_embeddings(mol)
        h_return = embeddings['text']
        h_attention_mask = mol['input']["attention_mask"]
        
        return h_return, h_attention_mask

    def get_enhanced_prompts(self, mol):
        text_embeddings = self.text_embeddings(mol)
        prompt_embeddings = self.prompt_dy_embeddings()
        prompt_loss = self.PromptSelectionModel(mol['type'], text_embeddings, prompt_embeddings)
        best_prompt_indices = self.PromptSelectionModel.get_best_prompt_indices(mol['type'], text_embeddings, prompt_embeddings)
        best_prompts = self.PromptManager.get_prompts_by_indices(best_prompt_indices)
        enhanced_prompts = []
        if self.config.reaction_type:
       
            assert len(best_prompts) == len(mol['type']), "The number of prompts and mol types must match."
            
            for prompt, mol_type in zip(best_prompts, mol['type']):
                enhanced_prompt = f"{prompt}{mol_type}"
                enhanced_prompts.append(enhanced_prompt)
        else:
            enhanced_prompts = best_prompts
        
        return enhanced_prompts
    
    def encode_h_dy(self, mol):
        text_embeddings = self.text_embeddings(mol)
        prompt_embeddings = self.prompt_dy_embeddings()
        prompt_loss = self.PromptSelectionModel(mol['type'], text_embeddings, prompt_embeddings)
        enhanced_prompts = self.get_enhanced_prompts(mol)
        #print(enhanced_prompts)
        mol['instruction'] = self.tokenizer(enhanced_prompts, padding=True, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        h_return, h_attention_mask = self.encode_h(mol)
        return prompt_loss, h_return, h_attention_mask
        
    def encode_embeddings(self, mol):
        embeddings = {}

        text_embeddings = self.text_encode(mol)
        embeddings['text'] = text_embeddings
        prompt_embeddings = self.prompt_encode(mol)
        embeddings['prompt'] = prompt_embeddings
        
        return embeddings
    
    def text_encode(self, mol):
        h = self.language_model.encode(mol['input'])
        return h
    
    def text_embeddings(self, mol):
        h = self.language_model.encode(mol['input'])
        h = h.transpose(1, 2) 
        h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        return h
    
    def output_encode(self, mol):
        h = self.language_model.encode(mol['output'])
        h = self.fc_2d(h)
        return h
    
    def output_embeddings(self, mol):
        h = self.language_model.encode(mol['output'])
        h = self.fc_2d(h)
        h = h.transpose(1, 2) 
        h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        return h
    
    def input_output_embeddings(self, mol):
        mol['input']['input_ids'] = torch.cat([mol['input']['input_ids'] , mol['output']['input_ids'] ], dim=1)
        mol['input']['attention_mask'] = torch.cat([mol['input']["attention_mask"], mol['output']["attention_mask"]], dim=1)
        h = self.language_model.encode(mol['input'])
        h = self.fc_2d(h)
        h = h.transpose(1, 2) 
        h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        return h
    
    def input_encode(self, mol):
        h = self.language_model.encode(mol['input'])
        h = self.fc_2d(h)
        return h
    def input_embeddings(self, mol):
        h = self.language_model.encode(mol['input'])
        h = self.fc_2d(h)
        h = h.transpose(1, 2) 
        h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        return h
    
    def prompt_encode(self, mol):
        h = self.language_model.encode(mol['instruction'])
        return h
    
    def prompt_encode_dy(self):
        self.prompts_tensors = self.prompts_tensor.to(self.config.device)
        h = self.language_model.encode(self.prompts_tensor)
        return h
    
    def prompt_dy_embeddings(self):
        self.prompts_tensors = self.prompts_tensor.to(self.config.device)
        h = self.language_model.encode(self.prompts_tensor)
        h = h.transpose(1, 2) 
        h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        return h
    
    def generate_text(self, mol):
        text_embeddings = self.text_embeddings(mol)
        prompt_embeddings = self.prompt_dy_embeddings()
        best_prompt_indices = self.PromptSelectionModel.get_best_prompt_indices(mol['type'], text_embeddings, prompt_embeddings)
        best_prompts = self.PromptManager.get_prompts_by_indices(best_prompt_indices)
        enhanced_prompts = []
        if self.config.reaction_type:
            assert len(best_prompts) == len(mol['type']), "The number of prompts and mol types must match."
            
            for prompt, mol_type in zip(best_prompts, mol['type']):
                enhanced_prompt = f"{prompt}{mol_type}"
                enhanced_prompts.append(enhanced_prompt)
        else:
            enhanced_prompts = best_prompts
        #print(enhanced_prompts)
        mol['instruction'] = self.tokenizer(enhanced_prompts, padding=True, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        h, h_attention_mask = self.encode_h(mol)
        h = BaseModelOutput(
                last_hidden_state= h,
                hidden_states= None,
                attentions= None
            )
        text = self.language_model.decode(
                encoder_outputs = h, 
                encoder_attention_mask = h_attention_mask, 
                num_beams = 5,
                max_length = 512
            )
        
        return text
    
    def process_prompts(self):

        task_type = self.config.task
        prompts_list = self.PromptManager.get_all_prompts_by_type(task_type)
        encoding = self.tokenizer(prompts_list, padding=True, return_tensors="pt", add_special_tokens=True)
        return encoding