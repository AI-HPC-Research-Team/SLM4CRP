# -*- coding: utf-8 -*-
from evaluations.mol_translation_metrics import mol_evaluate
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


def test_reactions(targets, preds, args):
    bleu_score, _meteor_score, exact_match_score, valid_score, similarity_score, result_dataframe= mol_evaluate(targets, preds)
    #print(result_dataframe)
    message = "input: {}, Metrics: bleu_score: {}, meter_score: {}, em-score: {}, similarity_score: {}, validity_score: {}".format(args.input_modal, bleu_score, _meteor_score, exact_match_score, similarity_score, valid_score)
    return message


def is_valid_smiles(smiles):
    """Check if a SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

