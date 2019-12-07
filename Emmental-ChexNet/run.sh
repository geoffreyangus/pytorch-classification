# python run.py \
#     --data_path $CXR8DATA \
#     --image_path $CXR8IMAGES \
#     --log_path log_abnorm_ml_slices \
#     --seed 0 \
#     --n_epochs 30 \
#     --train_split train \
#     --valid_split val \
#     --optimizer sgd \
#     --lr 0.01 \
#     --min_lr 1e-6 \
#     --counter_unit epoch \
#     --evaluation_freq 1 \
#     --checkpointing 1 \
#     --checkpoint_metric Abnormal/CXR8/val/accuracy:max \
#     --batch_size 16 \
#     --slices 0 \
#     --tasks TRIAGE \
#     --device 0 \
#     --dataparallel 1 \
#     --lr_scheduler linear

python run.py \
    --data_path $CXR8DATA \
    --image_path $CXR8IMAGES \
    --batch_size 16 \
    --log_path logs \
    --seed 0 \
    --n_epochs 20 \
    --train_split train \
    --valid_split val \
    --optimizer sgd \
    --lr 0.001 \
    --l2 0.0 \
    --lr_scheduler plateau \
    --lr_scheduler_step_unit epoch \
    --plateau_lr_scheduler_metric model/all/val/loss \
    --plateau_lr_scheduler_mode min \
    --plateau_lr_scheduler_factor 0.1 \
    --plateau_lr_scheduler_patience 0 \
    --counter_unit epoch \
    --evaluation_freq 1 \
    --checkpointing 1 \
    --checkpoint_metric model/all/val/loss:min \
    --slices 1 \
    --tasks TRIAGE  \
    --device 0 \
    --dataparallel 1 \
    # --max_data_samples 100\
