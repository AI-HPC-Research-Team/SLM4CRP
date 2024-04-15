import torch
import torch.nn as nn
from models import SUPPORTED_Model, SUPPORTED_CKPT, PromptManager
from models.momu import MoMu
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
from torch.nn.functional import one_hot


class MolModel(nn.Module):
    def __init__(self, config=None):
        super(MolModel, self).__init__()
        self.config = config
        self.config.hidden_size = 768
        self.language_model  = SUPPORTED_Model[self.config.model_name](self.config)
        self.fc_2d = nn.Linear(in_features=self.config.hidden_size, out_features=2)
        self.reaction_type = self.config.reaction_type
        self.mode = self.config.mode
        self.tokenizer = self.language_model.tokenizer
        self.criterion = nn.CrossEntropyLoss()
            
        
        if(self.config.freeze == True):
            for k, v in self.language_model.named_parameters():
                v.requires_grad = False
                
                
    def forward(self, mol):
        #h, h_attention_mask = self.encode_h(mol) 
        h, h_attention_mask = self.encode_h_label_prompt(mol) 
        labels = {}
        #labels["input_ids"] = mol[f"{self.output_modal}_labels"]["input_ids"].masked_fill(~mol[f"{self.output_modal}_labels"]["attention_mask"].bool(), -100)
        labels["input_ids"] = mol["label"]["input_ids"]
        labels["attention_mask"] = mol["label"]["attention_mask"]

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
        
        total_loss = task_loss
        
        return total_loss
    
    def encode_h(self, mol):
        embeddings = self.encode_embeddings(mol)
        #labels = mol[f"{self.output_modal}_labels"]
        h_return = torch.cat([embeddings['prompt'], embeddings['text']], dim=1)
        h_attention_mask = torch.cat([mol['instruction']["attention_mask"], mol['input']["attention_mask"]], dim=1)
        
        return h_return, h_attention_mask
    
    def encode_h_label_prompt(self, mol):
        embeddings = self.encode_embeddings(mol)
        #labels = mol[f"{self.output_modal}_labels"]
        h_return = torch.cat([embeddings['prompt'], embeddings['text']], dim=1)
        h_attention_mask = torch.cat([mol['prompt']["attention_mask"], mol['input']["attention_mask"]], dim=1)
        
        return h_return, h_attention_mask
    
    def encode_h_label(self, mol):
        embeddings = self.encode_embeddings(mol)
        h_return = embeddings['text']
        h_attention_mask = mol['input']["attention_mask"]
        
        return h_return, h_attention_mask


    def encode_embeddings(self, mol):
        embeddings = {}
        text_embeddings = self.text_encode(mol)
        embeddings['text'] = text_embeddings
        prompt_embeddings = self.prompt_encode(mol)
        embeddings['prompt'] = prompt_embeddings
        return embeddings
    
    def prompt_encode(self, mol):
        h = self.language_model.encode(mol['prompt'])
        return h
    
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
    
    def generate_text(self, mol):
        h, h_attention_mask = self.encode_h_label_prompt(mol) 
        h = BaseModelOutput(
                last_hidden_state= h,
                hidden_states= None,
                attentions= None
            )
        text = self.language_model.decode(
                encoder_outputs = h, 
                encoder_attention_mask = h_attention_mask, 
                num_beams = 5,
                max_length = 2
            )
        
        return text