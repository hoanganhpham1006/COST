# COST: Video Dialog as Conversation about Objects Living in Space-Time
## Prerequisites
pip install -r requirements.txt

## Preprocessing
Refer to preprocess/README.md

## Training
Multi-GPU Training
`python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py --config configs/mycfg.yml TRAIN.BATCH_SIZE 512 SYSTEM.NUM_WORKERS 32 OPT.BASE_LR 3e-3 OPT.WARMUP_EPOCHS 8 TRAIN.EPOCHS 50 EXPERIMENT myexp`

## Testing
Multi-GPU Testing
`python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py --config configs/mycfg.yml --resume weights/best_myexp.pth --test-only TRAIN.BATCH_SIZE 512 SYSTEM.NUM_WORKERS 32 OPT.BASE_LR 3e-3 OPT.WARMUP_EPOCHS 5 TRAIN.EPOCHS 50`

