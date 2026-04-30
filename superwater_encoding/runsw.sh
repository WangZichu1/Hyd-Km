#!/bin/bash

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

TARGET_ENV="superwater"

if ! command -v conda &> /dev/null; then
    echo "error: conda not found"
    exit 1
fi
eval "$(conda shell.bash hook)"

if [ "$CONDA_DEFAULT_ENV" != "$TARGET_ENV" ]; then
    echo "activate: $TARGET_ENV"
    conda activate "$TARGET_ENV"
else
    echo "already in env: $TARGET_ENV"
fi
python -c "import torch; print('Visible GPUs:', torch.cuda.device_count())"

file=$1

if [ -z "$file" ]; then
  echo "Usage: $0 <pdb_folder_name>"
  echo "Example: $0 testpdb10"
  echo "Note: The folder must be located inside the 'data' directory."
  exit 1
fi

# 1. Organize Dataset & Generate Split File
# This step automatically generates data/splits/${file}_organized.txt
python organize_pdb_dataset.py --raw_data $file --output_dir ${file}_organized

# 2. Prepare FASTA
python datasets/esm_embedding_preparation_water.py --data_dir data/${file}_organized --out_file data/prepared_for_esm_${file}_organized.fasta

cd data

# 3. Extract Embeddings
python ../esm/scripts/extract.py esm2_t33_650M_UR50D prepared_for_esm_${file}_organized.fasta ${file}_organized_embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096

cd ..

# 4. Inference
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m inference_water_pos_weight_1 \
    --original_model_dir workdir/all_atoms_score_model_res15_17092 \
    --confidence_dir workdir/confidence_model_17092_sigmoid_rr15 \
    --data_dir data/${file}_organized \
    --ckpt best_model.pt \
    --all_atoms \
    --cache_path data/cache_confidence \
    --split_test data/splits/${file}_organized.txt \
    --batch_size 1 \
    --inference_steps 20 \
    --esm_embeddings_path data/${file}_organized_embeddings_output \
    --cap 0.1 \
    --running_mode test \
    --mad_prediction \
    --save_pos \
    --water_ratio 15

# ==========================================
# 5. 
# ==========================================
rm -rf ./_____FILE/${file}
mkdir -p ./_____FILE/${file}
mkdir -p ./_____FILE/${file}/out

#   Embedding
mv ./inference_out/inferenced_pos_rr15_cap0.1/* ./_____FILE/${file}/out

# 
mv ./data/*${file}*.fasta ./_____FILE/${file}
mv ./data/${file}_organized ./_____FILE/${file}
mv ./data/${file}_organized_embeddings_output ./_____FILE/${file}
mv ./data/splits/*${file}*.txt ./_____FILE/${file}
