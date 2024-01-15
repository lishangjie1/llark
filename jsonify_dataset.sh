

export GOOGLE_CLOUD_PROJECT="/mnt/nas/users/lsj/music/llark"
export GCS_BUCKET_NAME="/mnt/nas/users/lsj/music/llark"
python scripts/preprocessing/jsonify_dataset.py \
    --dataset fma \
    --input-dir data/fma_metadata \
    --output-dir data/tmp \
    --split train \
