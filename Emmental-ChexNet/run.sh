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
    --log_path logs \
    --seed 0 \
    --n_epochs 1 \
    --train_split train \
    --valid_split val \
    --optimizer sgd \
    --lr 0.001 \
    --min_lr 1e-6 \
    --counter_unit epoch \
    --evaluation_freq 1 \
    --checkpointing 1 \
    --checkpoint_metric model/val/accuracy:max \
    --batch_size 16 \
    --slices 0 \
    --tasks CXR8 \
    --max_data_samples 10000 \
    --device 0 \
    --dataparallel 1
