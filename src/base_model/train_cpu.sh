accelerate launch --config_file accelerate_config.yaml run_recolor_training --train_batch_size 1 --val_batch_size 1 --learning_rate 0.0002 \
 --num_epochs 1000 --validation_interval 2
 