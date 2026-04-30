import os
import argparse

def generate_split_file(data_dir, output_txt):

    if not os.path.exists(data_dir):
        print(f"Error: data directory {data_dir} does not exist.")
        return


    pdb_ids = [
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ]
    
    pdb_ids.sort() # sort to ensure consistent order

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(output_txt, 'w') as f:
        for pid in pdb_ids:
            f.write(f"{pid}\n")
    
    print(f"Successfully generated file: {output_txt}")
    print(f"Total {len(pdb_ids)} PDB ID(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate split txt list file based on data directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to organized data directory (e.g.: data/testpdb_organized)")
    parser.add_argument("--output_txt", type=str, required=True, help="Output txt file path (e.g.: data/splits/testpdb.txt)")
    
    args = parser.parse_args()
    generate_split_file(args.data_dir, args.output_txt)