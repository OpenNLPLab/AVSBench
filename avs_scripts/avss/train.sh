
setting='AVSS'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"
gpu_num=1

spring.submit arun --gpu -n${gpu_num} --gres=gpu:${gpu_num}  --ntasks-per-node ${gpu_num} --quotatype=auto -p clever --job-name="train_${setting}_${visual_backbone}" \
"
python train.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --max_epoches 60 \
        --train_batch_size 4 \
        --val_batch_size 4 \
        --lr 0.0001 \
        --start_eval_epoch 15 \ 
        --eval_interval 2 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag \
        --masked_av_flag \
        --masked_av_stages 0 1 2 3 \
        --lambda_1 0.5 \
        --kl_flag \
"