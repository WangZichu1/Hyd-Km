import pandas as pd
import os
import re

def match_and_add_filenames(table_path, folder_path1, folder_path2, folder_path3, subindex, uniprot_col_name, output_path):
    """
        table_path
        folder_path
        uniprot_col_name
        output_path 
    """
    
    # 1
    if table_path.endswith('.csv'):
        df = pd.read_csv(table_path)
    elif table_path.endswith('.xlsx'):
        df = pd.read_excel(table_path,sheet_name="Sheet1")
    else:
        df = pd.read_table(table_path)
    
    print(f"row {len(df)} ")
    
    # 2
    all_files1 = [f for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f))]
    all_files2 = [f for f in os.listdir(folder_path2) if os.path.isfile(os.path.join(folder_path2, f))]
    all_files3 = [f for f in os.listdir(folder_path3) if os.path.isfile(os.path.join(folder_path3, f))]

    # 3
    def find_matching_filenames1(uniprot_id):
        if pd.isna(uniprot_id):
            return ""
        
        uniprot_str = str(uniprot_id).strip()
        if not uniprot_str:
            return ""
        
        
        matches1 = []

        for filename in all_files1:

            if uniprot_str.lower() in filename.lower():
                matches1.append(filename)
        

        if matches1:
            return "; ".join(matches1)
        else:
            return "not found"

    def find_matching_filenames2(uniprot_id):
        if pd.isna(uniprot_id):
            return ""
        
        uniprot_str = str(uniprot_id).strip()
        if not uniprot_str:
            return ""


        matches2 = []
        for filename2 in all_files2:

            if uniprot_str.lower() in filename2.lower():
                matches2.append(filename2)
        

        if matches2:
            return "; ".join(matches2)
        else:
            return "not found"

    def find_matching_filenames3(idx):
        if pd.isna(idx):
            return ""
        
        idx_str = str(idx).strip()
        
        if not idx_str:
            return ""
        
        matches3 = [
            f for f in all_files3
            if idx_str in f   
        ]
        #matches3 = [f for f in all_files3 if f == idx_str]   
        return "; ".join(matches3) if matches3 else "not found"    
    
    # 4
    df['full_esm'] = df[uniprot_col_name].apply(find_matching_filenames1)
    df['full_hyd'] = df[uniprot_col_name].apply(find_matching_filenames2)
    df['full_sub'] = df[subindex].apply(find_matching_filenames3)


    # 5
    matched_count = (df['full_esm'] != "not found").sum()
    print(f"found: {matched_count} ")
    print(f"not found: {len(df) - matched_count} ")
    
    # 6
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, sep='\t')
    
    print(f"saved: {output_path}")
    
    # 7
    print("\n5rows:")
    print(df.head().to_string())
    
    return df

# ====================
if __name__ == "__main__":
    # 
    TABLE_PATH = ""          
    FOLDER_PATH1 = "C:/Users/wangz/Desktop/hyd-kcat/esm_mean_embeddings"   
    FOLDER_PATH2 = ""
    FOLDER_PATH3 = ""
    UNIPROT_COL = ""                
    subindex = ""
    OUTPUT_PATH = ""  
    
    #
    result_df = match_and_add_filenames(
        TABLE_PATH, FOLDER_PATH1, FOLDER_PATH2, FOLDER_PATH3, subindex, UNIPROT_COL, OUTPUT_PATH
    )