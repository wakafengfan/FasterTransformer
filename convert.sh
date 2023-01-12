python3 examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir mt5-large/c-models \
        -in_file models/ \
        -inference_tensor_para_size 1 \
        -weight_data_type fp32