import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
from rdkit import Chem

def get_molT5_embed(smiles_list, Molt5_model):
#smiles_list：SMILES  ["CCO", "C=C"]
#final_values： (N, D) 
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model,
                                            #legacy=False,
                                            clean_up_tokenization_spaces=True#False
                                            )
    model = T5EncoderModel.from_pretrained(Molt5_model)
    N_smiles = len(smiles_list)

    if len(set(smiles_list)) == 1:#
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state# [1, seq_len, D]
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()#  drop[EOS] 
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)

    else:
        final_values = []
        for smile in smiles_list:
            input_ids = tokenizer(smile, return_tensors="pt").input_ids
            outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
            final_values.append(embed.reshape(1, -1))#

        final_values = np.concatenate(final_values, axis=0)
    
    return final_values

#test
# smiles = inp_df["SMILES"].values
