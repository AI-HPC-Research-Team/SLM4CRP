'''
Code from https://github.com/blender-nlp/MolT5
```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''


import pickle
import argparse
import csv

import os.path as osp

import numpy as np

# load metric stuff

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

from Levenshtein import distance as lev
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import MACCSkeys
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def check_validity(smiles_list):

    for smiles in smiles_list:
        if not Chem.MolFromSmiles(smiles):
            return False
    return True

def calculate_group_similarity(targets, preds, fingerprint_type='morgan', distance_metric='tanimoto'):

    def get_fingerprint(mol, fingerprint_type):

        if fingerprint_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        elif fingerprint_type == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        elif fingerprint_type == 'rdk':
            return Chem.RDKFingerprint(mol)
        else:
            raise ValueError("Unsupported fingerprint type: {}".format(fingerprint_type))
    
    def calculate_similarity(fps1, fps2, distance_metric):
  
        total_similarity = 0
        for fp1 in fps1:
            max_similarity = 0
            for fp2 in fps2:
                if distance_metric == 'tanimoto':
                    similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
                elif distance_metric == 'dice':
                    similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.DiceSimilarity)
                else:
                    raise ValueError("Unsupported distance metric: {}".format(distance_metric))
                max_similarity = max(max_similarity, similarity)
            total_similarity += max_similarity
        return total_similarity / len(fps1)
    
   
    mol_targets = [Chem.MolFromSmiles(smiles) for smiles in targets if Chem.MolFromSmiles(smiles) is not None]
    mol_preds = [Chem.MolFromSmiles(smiles) for smiles in preds if Chem.MolFromSmiles(smiles) is not None]
    
   
    fps_targets = [get_fingerprint(mol, fingerprint_type) for mol in mol_targets]
    fps_preds = [get_fingerprint(mol, fingerprint_type) for mol in mol_preds]
    

    similarity = calculate_similarity(fps_targets, fps_preds, distance_metric)
    return similarity

def mol_evaluate(targets, preds, verbose=False):
    outputs = []

    for i in range(len(targets)):
            gt_smi = targets[i]
            ot_smi = preds[i]
            outputs.append((gt_smi, ot_smi))


    bleu_scores = []
    meteor_scores = []
    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')


        gt_tokens = [c for c in gt]

        out_tokens = [c for c in out]

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose: print('BLEU score:', bleu_score)

    # Meteor score
    _meteor_score = np.mean(meteor_scores)
    # print('Average Meteor score:', _meteor_score)

    rouge_scores = []

    references = []
    hypotheses = []

    levs = []

    num_exact = 0

    bad_mols = 0

    result_dataframe = pd.DataFrame(outputs, columns=['targets', 'preds'])
    result_dataframe['exact'] =0
    result_dataframe['valid'] =0
    result_dataframe['similarity'] = 0.0
    result_dataframe['QED'] = 0
    #result_dataframe['lev'] =100000
    #print(result_dataframe.head())
    for index, row in tqdm(result_dataframe.iterrows(), total=result_dataframe.shape[0]):
        valid = 0
        exact = 0
        targets = row['targets'].split('.')  
        preds = row['preds'].split('.')


        if set(targets) == set(preds):
            result_dataframe.at[index, 'exact'] = 1
            exact = 1
   
        if check_validity(preds):
            result_dataframe.at[index, 'valid'] = 1
            valid = 1
       
        if valid == 1 and exact != 1:
            similarity = calculate_group_similarity(targets, preds, fingerprint_type='morgan', distance_metric='tanimoto')
            #print(similarity)
            result_dataframe.at[index, 'similarity'] = similarity
        elif exact == 1:
            similarity = 1.0
            #print(similarity)
            result_dataframe.at[index, 'similarity'] = similarity


    exact_match_score = result_dataframe['exact'].sum() / len(result_dataframe)
    #print(f"Exact Match Score: {exact_match_score}")


    valid_score = result_dataframe['valid'].sum() / len(result_dataframe)
    #print(f"Valid Score: {valid_score}")

    valid_similarity_rows = result_dataframe[result_dataframe['valid'] == 1]
    similarity_score = valid_similarity_rows['similarity'].sum() / len(valid_similarity_rows)

    return bleu_score, _meteor_score, exact_match_score, valid_score, similarity_score, result_dataframe