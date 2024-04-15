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
from models.model_manager_label import MolModel
from datasets.dataset_manager_label import MolDataset
from utils.xutils import print_model_info, custom_collate_fn, ToDevice
from transformers import T5Tokenizer
from evaluations.metric import test_reactions
from torch.optim.lr_scheduler import StepLR
from utils import AverageMeter
import datetime
import time
from sklearn.cluster import KMeans


def preprocess_result_list(input_list):
    # 预处理列表，尝试将文本转换为数字，无法转换的标记为113
    processed_list = []
    for item in input_list:
        try:
            processed_list.append(int(item))
        except ValueError:
            # 如果转换失败，则添加特定的标记值
            processed_list.append(0)
    return processed_list

def task_dataset(args):

    if(args.dataset_toy == 'toy'):
        args.dataset_folder = args.dataset_folder+'toy/'
            
    if(args.task=='forward'):
        args.dataset = args.dataset_folder+'forward_reaction_prediction.json'
    elif(args.task=='retro'):
        args.dataset = args.dataset_folder+'retrosynthesis.json'
    elif(args.task=='reagent'):
        args.dataset = args.dataset_folder+'reagent_prediction.json'
    elif(args.task=='reactions'):
        args.dataset = args.dataset_folder+'reactions.json'
        
    return args

def train_mol_label(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss = None):
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
            loss.backward()
            optimizer.step()
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
            if not os.path.exists(f"{args.ckpt_output_path}/label/{args.task}"):
                os.makedirs(f"{args.ckpt_output_path}/label/{args.task}")
  
            ckpt_file = f"{args.cluster_number}_cluster_{epoch}_{timestamp}.pth"
            ckpt_path = os.path.join(f"{args.ckpt_output_path}/label/{args.task}", ckpt_file)
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
            metric = test_mol_decoder(test_loader, model, device, args, message)
            message = message + f"epoch {epoch-1} metric : {metric}."
            print(message)
            print(loss_values)
            print("Early stopping due to reaching patience limit.")
            break
            
def train_mol_label_one_epoch(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    last_ckpt_file = None
    patience = 0
    device = args.device
    logger.info("Training...")
    train_loss = []
    train_loader = tqdm(train_loader, desc="Training")
    for mol in train_loader:
        mol = ToDevice(mol, args.device)
        loss = model(mol)  
        loss.backward()
        optimizer.step()
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
    message = f"best_loss: {best_loss} ,loss_values: {loss_values}"
    print(message)
    
    return model, best_loss, val_loss

def data_train_label_one_epoch(dataset, model, optimizer, scheduler, args, best_loss = None):
    df = dataset.return_df()
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                             pin_memory=True)
    data_loader = tqdm(data_loader, desc="Labeling")
    label_vector_list = []
    for i, batch in enumerate(data_loader):
        mol = ToDevice(batch, args.device)
        label_vectors = vector_generate(model, mol, args)    
            #print(f"label_vector {i} : {label_vectors}")
        for x in label_vectors:
            label_vector_list.append(x)
            # if i >= 1:
            #     break
    assert len(label_vector_list) == len(df), "The length of the list must match the number of rows in the DataFrame."
    difference_vectors = np.array(label_vector_list)
    if(args.vector_type == 3):
        X = np.array(difference_vectors).reshape(-1, 1)
        kmeans = KMeans(n_clusters=args.cluster_number, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
    else:
        kmeans = KMeans(n_clusters=args.cluster_number, random_state=42)
        cluster_labels = kmeans.fit_predict(difference_vectors)
    df['reaction_label'] = cluster_labels
        #print(df['reaction_label'].head())
    args.reaction_type = True
    train_dataset = MolDataset(df = df, split = "train",
                               tokenizer_org = tokenizer_org,
                               args = args
                              )
    valid_dataset = MolDataset(df = df, split = "valid",
                               tokenizer_org = tokenizer_org,
                               args = args
                              )
    test_dataset = MolDataset(df = df, split = "test",
                              tokenizer_org = tokenizer_org,
                              args = args
                             )
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        
    model, best_loss, val_loss= train_mol_label_one_epoch(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
    
    return model, best_loss, val_loss, train_dataset, valid_dataset, test_dataset

def val_mol_decoder(valid_loader, model, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            truth = mol['type']
            mol = ToDevice(mol, device)
            loss = model(mol)
            if(i==0):
                result = model.generate_text(mol)
                result = preprocess_result_list(result)
                print(f"truth:{truth} | Result : {result}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)

def __eval_multi_label(truth_list, result_list, n_classes):
    result_list = preprocess_result_list(result_list)
    truth_list_binarized = label_binarize(truth_list, classes=list(range(n_classes)))
    result_list_binarized = label_binarize(result_list, classes=list(range(n_classes)))

    print(f"truth_list_binarized: {truth_list_binarized}")
    print(f"result_list_binarized: {result_list_binarized}")
    #roc_auc = roc_auc_score(truth_list_binarized, result_list_binarized, average='macro', multi_class='ovr')
    #print(f"ROC AUC Score: {roc_auc}")
    

    accuracy = accuracy_score(truth_list, result_list)
    print(f"Accuracy: {accuracy}")
    
    precision = precision_score(truth_list, result_list, average='macro')
    recall = recall_score(truth_list, result_list, average='macro')
    f1 = f1_score(truth_list, result_list, average='macro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return accuracy, precision, f1 , recall

def test_mol_decoder(test_loader, model, device, args, message = None):
    model.eval()
    test_loss = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logger.info("Testing...")
    if not os.path.exists(f"{args.result_save_path}/{args.task}/label"):
        os.makedirs(f"{args.result_save_path}/{args.task}/label")
    result_file = f"{args.result_save_path}/{args.task}/label/{args.cluster_number}.txt"
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        truth_list = []
        result_list = []
        for mol in test_loader:
            truth = mol['type']
            if(type(truth) == 'int'):
                truth_list = truth_list + [truth]
            else:
                truth_list = truth_list + truth
            mol = ToDevice(mol, device)
            result = model.generate_text(mol)
            result = preprocess_result_list(result)
            if(i==0):
                print(f"truth:{truth[0]} | Result : {result[0]}")
            i=i+1
            result_list = result_list + result
        assert len(truth_list) == len(result_list)
        acc, pre_score, f1, recall = __eval_multi_label(truth_list, result_list, args.cluster_number)
        message_1 = f"dataset metric 1: acc: {acc}, pre_score : {pre_score}, f1 : {f1}, recall: {recall}, loss: {message}"
        print(message_1)
        with open(result_file, 'a') as f:   
            f.write(str(args) + "\n")
            if(message == None):
                pass
            else:
                f.write(message + "\n") 
            f.write(timestamp+':'+message_1 + "\n") 
        
        return message_1

def result_list_RT(test_loader, model, device, args, message = None):
    model.eval()
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        result_list = []
        vector_list = []
        for mol in test_loader:
            mol = ToDevice(mol, device)
            result = model.generate_text(mol)
            result = preprocess_result_list(result)
            vector = vector_generate(model, mol, args)
            print(vector)
            for v in vector:
                vector_list.append(vector)
            if(i==0):
                print(f"Result : {result[0]}")
            i=i+1
            for r in result:
                result_list.append(r)
        return result_list, vector_list
    
def vector_generate(model, mol, args):
    if(args.vector_type == 0):
        output_tensor = model.output_embeddings(mol)
        vector = output_tensor.cpu().detach().numpy()
    elif(args.vector_type == 1):
        input_tensor = model.input_embeddings(mol)
        output_tensor = model.output_embeddings(mol)
        input_np = input_tensor.cpu().detach().numpy()
        output_np = output_tensor.cpu().detach().numpy()
        vector = output_np - input_np
    elif(args.vector_type == 2):
        input_output_tensor = model.input_output_embeddings(mol)
        vector = input_output_tensor.cpu().detach().numpy()
    elif(args.vector_type == 3):
        input_tensor = model.input_embeddings(mol)
        output_tensor = model.output_embeddings(mol)
        input_np = input_tensor.cpu().detach().numpy()
        output_np = output_tensor.cpu().detach().numpy()
        vector = np.einsum('ij,ij->i', output_np, input_np)
        
    return vector

def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--task", type=str, default="forward")
    parser.add_argument('--dataset_toy', type=str, default='normal')
    parser.add_argument("--dataset_folder", type=str, default='../datasets/Mol/SMILES/')
    parser.add_argument("--ckpt_output_path", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--reaction_type", type=bool, default=False)
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
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--vector_type", type=int, default=2, help='0/1/2/3')
    parser.add_argument("--cluster_number", type=int, default=10, help='3-12')
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
        args.dataset = f"{args.dataset_folder}type/toy/type_test/nn_{args.task}.json"
        model = MolModel(args)
        print_model_info(model,level=2)
        tokenizer_org = T5Tokenizer.from_pretrained(args.model_pretrain)
        print(args.task)

        model.to(args.device)
        
        dataset = MolDataset(split = "all",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
 
        data_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)
        #print(test_dataset[1])
        best_loss = None
        best_loss_list = []
        val_loss_list = []
        best_train_dataset = None
        best_valid_dataset = None
        best_test_dataset = None
        patience = 0
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        for epoch in range(args.epochs):
            model, best_loss, val_loss, train_dataset, valid_dataset, test_dataset = data_train_label_one_epoch(dataset, model, optimizer, scheduler, args, best_loss)
            best_loss_list.append(best_loss)
            val_loss_list.append(val_loss)
            if(best_loss==val_loss):
                best_train_dataset = train_dataset
                best_valid_dataset = valid_dataset
                best_test_dataset = test_dataset
            elif(patience < args.patience):
                patience = patience + 1
                print(f"patience : {patience}")
            else:
                print("Early stopping due to reaching patience limit.")
                break
                
            print(f"best_loss_list : {best_loss_list}")
            print(f"val_loss_list : {val_loss_list}")
            
        train_loader = DataLoader(best_train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)
        valid_loader = DataLoader(best_valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(best_test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
  
        train_mol_label(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)

        
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
        
        tokenizer_org = T5Tokenizer.from_pretrained(args.model_pretrain)
        
        model = MolModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        model.to(args.device)
        logger.info("Loading model successed")
        
        # dataset
        args.dataset = f"{args.dataset_folder}type/type_test/nn_{args.task}.json"
        dataset = MolDataset(split = "all",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
 
        data_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)
        for number in range(2):
            args.cluster_number = 10-number*4
            logger.info("======== cluster_number %d ========" % (args.cluster_number))
            #print(test_dataset[1])
            best_loss = None
            best_loss_list = []
            val_loss_list = []
            best_train_dataset = None
            best_valid_dataset = None
            best_test_dataset = None
            patience = 0
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
            for epoch in range(args.epochs):
                model, best_loss, val_loss, train_dataset, valid_dataset, test_dataset = data_train_label_one_epoch(dataset, model, optimizer, scheduler, args, best_loss)
                best_loss_list.append(best_loss)
                val_loss_list.append(val_loss)
                if(best_loss==val_loss):
                    best_train_dataset = train_dataset
                    best_valid_dataset = valid_dataset
                    best_test_dataset = test_dataset
                elif(patience < args.patience):
                    patience = patience + 1
                    print(f"patience : {patience}")
                else:
                    print("Early stopping due to reaching patience limit.")
                    break

                print(f"best_loss_list : {best_loss_list}")
                print(f"val_loss_list : {val_loss_list}")

            train_loader = DataLoader(best_train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                      pin_memory=True)
            valid_loader = DataLoader(best_valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
            test_loader = DataLoader(best_test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)

            train_mol_label(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
            args.dataset = f"{args.dataset_folder}type/type_test/nn_valid_{args.task}.json"
            anno_valid_dataset = MolDataset(split = "valid",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
            args.dataset = f"{args.dataset_folder}type/type_test/nn_test_{args.task}.json"
            anno_test_dataset = MolDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
            df_valid = anno_valid_dataset.return_df()
            df_test = anno_test_dataset.return_df()
            anno_valid_loader = DataLoader(anno_valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
            anno_test_loader = DataLoader(anno_test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
            valid_result_list, valid_vector_list = result_list_RT(anno_valid_loader, model, args.device, args, message = None)
            test_result_list, test_vector_list = result_list_RT(anno_test_loader, model, args.device, args, message = None)


            df_valid['result'] = valid_result_list
            df_valid['vector'] = valid_vector_list

            df_test['result'] = test_result_list
            df_test['vector'] = test_vector_list
            df_train_1 = best_train_dataset.return_df()
            df_train_2 = best_valid_dataset.return_df()
            df_train_3 = best_test_dataset.return_df()
            # 合并df_train_1, df_train_2, df_train_3为df_train
            df_train = pd.concat([df_train_1, df_train_2, df_train_3], ignore_index=True)

            # 将DataFrames保存为CSV文件
            df_train.to_csv(f'df_{args.task}_{args.cluster_number}_train.csv', index=False)
            df_valid.to_csv(f'df_{args.task}_{args.cluster_number}_valid.csv', index=False)
            df_test.to_csv(f'df_{args.task}_{args.cluster_number}_test.csv', index=False)      
            
    if(args.mode == 'eval'):
        args.dataset = f"{args.dataset_folder}type/toy/type_test/nn_{args.task}.json"
        print(args.task)
        #args.latest_checkpoint = f"{args.latest_checkpoint}/{args.task}/text_8_20240314-1406.pth"
        args.reaction_type = True
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
        test_mol_decoder(test_loader, model, args.device, args)

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
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers = 8, collate_fn=custom_collate_fn)
        
        for i, batch in enumerate(train_loader):
            print(f"batch {i}")
        
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            else:
                print(f"batch {i} : {batch}")