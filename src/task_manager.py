# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from models.chemT5 import ChemT5
from models.model_manager import MolModel
from datasets.dataset_manager import MolDataset
from utils.xutils import print_model_info, custom_collate_fn, ToDevice
from transformers import T5Tokenizer
from evaluations.metric import test_reactions
from torch.optim.lr_scheduler import StepLR
from utils import AverageMeter
import datetime
import time
from sklearn.cluster import KMeans

def task_dataset(args):
 
    if(args.dataset_toy == 'toy'):
        args.dataset_folder = args.dataset_folder+'toy/'
            
    if(args.task=='forward'):
        args.dataset = args.dataset_folder+str(args.N)+'/forward_reaction_prediction.json'
    elif(args.task=='retro'):
        args.dataset = args.dataset_folder+str(args.N)+'/retrosynthesis.json'
    elif(args.task=='reagent'):
        args.dataset = args.dataset_folder+str(args.N)+'/reagent_prediction.json'
    elif(args.task=='reactions'):
        args.dataset = args.dataset_folder+str(args.N)+'/reactions.json'
        
    return args

def train_mol_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    last_ckpt_file = None
    patience = 0
    device = args.device
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for mol in train_loader:
            mol = ToDevice(mol, args.device)
            loss = model(mol)
            #accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()
        loss_values["train_loss"].append(np.mean(train_loss))
        val_loss = val_mol_decoder(valid_loader, model, device)
        test_loss = val_mol_decoder(test_loader, model, device)
        loss_values["val_loss"].append(val_loss)
        loss_values["test_loss"].append(test_loss)

        if best_loss == None or val_loss<best_loss :
            patience = 0
            best_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(f"{args.ckpt_output_path}/{args.task}"):
                os.makedirs(f"{args.ckpt_output_path}/{args.task}")
  
            ckpt_file = f"{args.input_modal}_{epoch}_{timestamp}.pth"
            ckpt_path = os.path.join(f"{args.ckpt_output_path}/{args.task}", ckpt_file)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss
            }, ckpt_path)
                
            message = f"epoch: {epoch}, best_loss:{best_loss} ,val_loss:{val_loss}, {ckpt_file} saved. "
            print(message)
            if last_ckpt_file is not None and os.path.exists(last_ckpt_file):
                os.remove(last_ckpt_file)
                print(f"Deleted checkpoint file: {last_ckpt_file}")
            last_ckpt_file = ckpt_path
            print(loss_values)
        else:
            patience = patience+1
            scheduler.step()
            if last_ckpt_file is not None :
                state_dict = torch.load(last_ckpt_file, map_location='cpu')["model_state_dict"]
                best_loss = torch.load(last_ckpt_file, map_location='cpu')["best_loss"]
                model.load_state_dict(state_dict, strict = False)
            #metric = test_mol_decoder(test_loader, model, device, message)
            #message = message + f"epoch {epoch-1} metric : {metric}."
            #print(message)
            print(loss_values)
        if patience > args.patience:
            message = f"epoch: {epoch}, best_loss: {best_loss} ,val_loss: {val_loss}, ckpt passed, patience : {patience}. "
            metric = test_mol_decoder(test_loader, model, device, message)
            message = message + f"epoch {epoch-1} metric : {metric}."
            print(message)
            print(loss_values)
            print("Early stopping due to reaching patience limit.")
            break

def val_mol_decoder(valid_loader, model, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            task = mol['task']
            truth = mol['truth']
            del mol['task']
            del mol['truth']
            mol = ToDevice(mol, device)
            loss = model(mol)
            if(i==1):
                result = model.generate_text(mol)
                print(f"{task[0]}:{truth[0]} | Result : {result[0]}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)


def test_mol_decoder(test_loader, model, device, message = None):
    model.eval()
    test_loss = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logger.info("Testing...")
    if not os.path.exists(f"{args.result_save_path}/{args.task}"):
        os.makedirs(f"{args.result_save_path}/{args.task}")
    result_file = f"{args.result_save_path}/{args.task}/{args.input_modal}.txt"
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        truth_list = []
        result_list = []
        for mol in test_loader:
            #mol = ToDevice(mol, device)
            task = mol['task']
            truth = mol['truth']
            truth_list = truth_list + truth
            del mol['task']
            del mol['truth']
            mol = ToDevice(mol, device)
            result = model.generate_text(mol)
            if(i==1):
                prompts = model.get_enhanced_prompts(mol)
                print(f"{task[0]}:{truth[0]} | prompt: {prompts[0]} |Result : {result[0]}")
            i=i+1
            result_list = result_list + result
        metric = test_reactions(truth_list, result_list, args)
        #metric = result_list
        print(metric)
        with open(result_file, 'a') as f:   
            f.write(str(args) + "\n")
            if(message == None):
                pass
            else:
                f.write(message + "\n") 
            f.write(timestamp+':'+metric + "\n") 
        
        return metric
        
def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--task", type=str, default="forward")
    parser.add_argument('--dataset_toy', type=str, default='normal')
    parser.add_argument("--dataset_folder", type=str, default='../datasets/Mol/SMILES/type/')
    parser.add_argument("--ckpt_output_path", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--input_modal", type=str, default="text")
    parser.add_argument("--reaction_type", type=bool, default=False)
    parser.add_argument("--moe", type=bool, default=False)
    parser.add_argument("--scaffold_split", type=bool, default=False)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default='chemt5')
    parser.add_argument("--model_output_path", type=str, default="../output")
    parser.add_argument("--log_save_path", type=str, default="../log")
    parser.add_argument("--result_save_path", type=str, default="../result")
    parser.add_argument("--latest_checkpoint", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_pretrain", type=str, default="../ckpts/text_ckpts/ChemT5-base-augm")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=300)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    args = task_dataset(args)
    print(args)

    if(args.mode == 'encoder_check'):
        model = MolModel(args)
        print_model_info(model,level=2)
        tokenizer_org = T5Tokenizer.from_pretrained(args.model_pretrain)
        print(args.task)
        test_dataset = MolDataset(split = "train",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        #print(test_dataset[1])
        print(f"dataset length {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        model.to(args.device)
        for i, batch in enumerate(test_loader):
            #print(f"batch {i} : {batch}")   
            batch = ToDevice(batch, args.device)
            print(f"batch_loss {i} : {model(batch)}")
            if i >= 1:
                break
                
    if(args.mode == 'train'):
        print(args.task)
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # model
        logger.info("Loading model ......")
        model = MolModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        # dataset
        logger.info("Loading dataset ......")

        tokenizer_org = T5Tokenizer.from_pretrained(args.model_pretrain)
        train_dataset = MolDataset(split = "train",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        valid_dataset = MolDataset(split = "valid",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        test_dataset = MolDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        #training
        #accelerator = Accelerator()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        #if(args.device == ""):
            #args.device = accelerator.device
        model = model.to(args.device)
        
        print(f"now device is {args.device}")
        
        train_mol_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
        
    if(args.mode == 'eval'):
        print(args.task)
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # model
        logger.info("Loading model ......")
        model = MolModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        # dataset
        logger.info("Loading dataset ......")

        tokenizer_org = T5Tokenizer.from_pretrained(args.model_pretrain)
        test_dataset = MolDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        model = model.to(args.device)
        print(f"now device is {args.device}")
        test_mol_decoder(test_loader, model, args.device)

    if(args.mode == 'model_check'):
        model = MolModel(args)
        print_model_info(model,level=2)
        
    if(args.mode == 'data_check'):
        tokenizer_org = T5Tokenizer.from_pretrained(args.model_pretrain)
        print(args.task)
        train_dataset = MolDataset(split = "train",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        test_dataset = MolDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        #print(test_dataset[1])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        
        for i, batch in enumerate(train_loader):
            print(f"batch {i}")
        
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            else:
                print(f"batch {i} : {batch}")