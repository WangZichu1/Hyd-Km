import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5EncoderModel


class MolT5Embedder:
    """
    MolT5 
    - save token embeddings
    """
    
    def __init__(self, model_name="laituan245/molt5-base", device=None):
        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"device: {self.device}")
        
        #tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=True
        )
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # infere，no dropout
        
        self.hidden_dim = self.model.config.d_model  # 768 for base
        
    def encode(self, smiles_list, return_mean=True, remove_eos=True):
        """
        SMILES
        
        Args:
            smiles_list: SMILES 
            return_mean: True  (N, D), False  (N, L, D)
            remove_eos:  </s> token
        
        Returns:
             return_mean=True:  (N, hidden_dim)  numpy 
             return_mean=False: list of (seq_len, hidden_dim) tensor
        """
        # batch tokenize
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,      # padding batch length
            truncation=True    # cut
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (N, L, D)
        
        if return_mean:
            # 
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (N, L, 1)
            
            if remove_eos:
                #  </s>  mask  0
                eos_mask = (input_ids == self.tokenizer.eos_token_id).float()
                mask_expanded = mask_expanded * (1 - eos_mask.unsqueeze(-1))
            
            sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)  # (N, D)
            counts = mask_expanded.sum(dim=1).clamp(min=1)             # (N, 1)
            embeddings = (sum_embeddings / counts).cpu().numpy()       # (N, D)
            return embeddings
        else:
            # 
            results = []
            for i, smile in enumerate(smiles_list):
                length = attention_mask[i].sum().item()  # include </s>
                seq = last_hidden[i, :length, :].cpu()   # (length, D)
                
                if remove_eos and length > 0:
                    seq = seq[:-1, :]  # drop </s>
                
                results.append(seq)
            return results
    
    def encode_single(self, smiles, return_mean=True, remove_eos=True):
        """single SMILES"""
        results = self.encode([smiles], return_mean=return_mean, remove_eos=remove_eos)
        return results[0]
    
    def __del__(self):
        """clear GPU """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_individual_embeddings(
    inp_fpath,
    out_dir,
    model_name="",
    device=None,
    id_col="IDs",
    smiles_col="Smiles",
    name_col="Substrate",
    return_mean=False,      
    remove_eos=True,
    batch_size=64
):
    """
    Args:
        inp_fpath:
        out_dir: 
        model_name: 
        device:
        id_col: ID 
        smiles_col: 
        name_col: 
        return_mean: True
        remove_eos: 
        batch_size:
    """
    # 
    os.makedirs(out_dir, exist_ok=True)
    
    # 
    embedder = MolT5Embedder(model_name=model_name, device=device)
    
    # 
    df = pd.read_excel(inp_fpath,sheet_name="600L-mutant")
    #df = pd.read_csv(inp_fpath)
    total = len(df)
    
    print(f" {total} , batch {batch_size}")
    print(f"save: {'mean' if return_mean else 'seq'}")
    print("-" * 50)
    
    saved_count = 0
    error_count = 0
    
    #
    for i in range(0, total, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_smiles = batch_df[smiles_col].tolist()
        batch_ids = batch_df[id_col].tolist()
        batch_names = batch_df[name_col].tolist()
        
        try:
            #
            embeddings = embedder.encode(
                batch_smiles,
                return_mean=return_mean,
                remove_eos=remove_eos
            )
            
            #
            for idx, (emb, sid, smile, name) in enumerate(zip(embeddings, batch_ids, batch_smiles, batch_names)):
                #clean
                safe_id = str(sid).replace('/', '_').replace('//', '_').replace(':', '_')
                out_path = os.path.join(out_dir, f"{safe_id}_{name}.pt")
                
                #
                save_dict = {
                    "id": sid,
                    "smiles": smile,
                    "name": name,
                    "embedding": emb,  # numpy array if mean, torch tensor if sequence
                    "shape": emb.shape if hasattr(emb, 'shape') else emb.size(),
                    "model": model_name,
                    "pooled": return_mean
                }
                
                torch.save(save_dict, out_path)
                saved_count += 1
            
            #
            current = min(i + batch_size, total)
            if current % 256 == 0 or current == total:
                print(f"run: {current}/{total} ({current/total*100:.1f}%)")
                
        except Exception as e:
            error_count += len(batch_df)
            print(f"batch {i}-{i+batch_size} fail: {e}")
            
            continue
    
    print("-" * 50)
    print(f"done: {saved_count}, fail: {error_count}")
    print(f"saved: {os.path.abspath(out_dir)}")
    
    return saved_count, error_count


# ========================================run

if __name__ == "__main__":
    

    save_individual_embeddings(
        inp_fpath="",
        out_dir="",
        return_mean=True,      #  (seq_len, 768) 
        remove_eos=True,        #  </s>
        batch_size=64
    )