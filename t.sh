python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --visible_devices=4,5,6,7 \
--model deit_tiny_patch16_224  --server-model deit_base_patch16_224 --batch-size 1024 \
--data-path /home/dataset/ImageNet --input-size 224 \
--output_dir ./images/ --output_list 105,522,928,1200,9818,10697,14476 \
--attention_mode mean --masking_mode attention_sum_threshold --masking_th 0.97 \
--uncer_mode min_entropy --uncer_th 1.0