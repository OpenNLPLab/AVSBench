
setting='S4'
visual_backbone="pvt" # "resnet" or "pvt"

spring.submit arun --gpu -n1 --gres=gpu:1 --quotatype=auto -p MMG --job-name="train_${setting}_${visual_backbone}" \
"
python train.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag 
"
