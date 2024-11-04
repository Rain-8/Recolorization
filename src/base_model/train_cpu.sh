accelerate launch --config_file "../common_utils/configs/accelerate_cpu_config.yaml" run_recolor_training.py --train_batch_size 1 --val_batch_size 1 --learning_rate 0.0002 \
 --num_epochs 1000 --validation_interval 2 --train_data_path "../../datasets/processed_data_v3/recolorization_data.json" --val_data_path "../../datasets/processed_data_v5/recolorization_data.json" 
 