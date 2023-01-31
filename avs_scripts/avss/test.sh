setting='AVSS'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"

spring.submit arun --gpu -n1 --gres=gpu:1 --ntasks-per-node 1 --quotatype=auto -p clever --job-name="test_${setting}_${visual_backbone}" \
"
python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "./train_logs/.../checkpoints/V2_pvt_miou_best.pth" \
    --test_batch_size 4 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    # --save_pred_mask
"

