# original
python run_with_submitit.py \
    --dataset_config configs/pretrain.json \
    --ngpus 4 \
    --nodes 8 \
    --ema \
    --backbone timm_tf_efficientnet_b3_ns

# effnetb3 - bert-s - detr-s
python run_with_submitit.py \
    --dataset_config configs/pretrain.json \
    --ema \
    --backbone timm_tf_efficientnet_b3_ns \
    --text_encoder_type google/bert_uncased_L-4_H-512_A-8 \
    # --enc_layers 3 \
    # --dec_layers 3 \
    # --dim_feedforward 1024 \
    # --nheads 4 \
    --batch_size 4 \
    --ngpus 4 \
    --nodes 4

# detr-s 589896 600224
# detr 658295

# vit-s - bert-s - detr-s
python run_with_submitit.py \
    --dataset_config configs/pretrain.json \
    --ema \
    --backbone timm_vit_small_patch16_384 \
    --text_encoder_type google/bert_uncased_L-4_H-512_A-8 \
    --enc_layers 3 \
    --dec_layers 3 \
    --dim_feedforward 1024 \
    --nheads 4 \
    --batch_size 4 \
    --ngpus 4 \
    --nodes 4 \
    --resume $WORK/mdetr_models/600226/checkpoint.pth \
    --constraint v100-32g

# 591325 600226 601805

# vit-s - bert-s - detr
python run_with_submitit.py \
    --dataset_config configs/pretrain.json \
    --ema \
    --backbone timm_vit_small_patch16_384 \
    --text_encoder_type google/bert_uncased_L-4_H-512_A-8 \
    --ngpus 4 \
    --nodes 8 \
    --resume $WORK/mdetr_models/626123/checkpoint.pth

# roberta 626103 645041
# bert-s 626123 644998
