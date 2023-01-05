from __future__ import print_function
import argparse
import json
from pathlib import Path
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import math
import datetime


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.bojone_tokenizers import SpTokenizer


def read_examples():
    data = []
    dic_data = json.load(Path('words_pred_log_69.json').open())
    for dic in dic_data:
        data.append(list(dic.values()))
    return data



def sequence_padding(inputs, length=None, padding=0, mode='post', with_mask=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    output_masks = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        m = np.pad([1]*len(x), pad_width, 'constant', constant_values=padding)
        output_masks.append(m)
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    if with_mask:
        return np.array(outputs), np.array(output_masks)

    return np.array(outputs)

    
class WordsGenDataset(Dataset):
    def __init__(self, args, examples, tokenizer) -> None:
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return self.examples[index]
    
    def __len__(self):
        return len(self.examples)
    
    def collate_fn(self, batch_samples):
        # batch_decoder_ids, batch_encoder_ids = [], []
        batch_encoder_ids = []
        batch_texts = []
        for context, t_ans, p_ans, s in batch_samples:
            c_token_ids, _ = self.tokenizer.encode('context: ' + context, maxlen=self.args.max_seq_len-20)
            q_token_ids, _ = self.tokenizer.encode('question: 这句的关键词是什么?')
            # a_token_ids, _ = self.tokenizer.encode(ans, maxlen=self.args.max_question_len)

            batch_encoder_ids.append(c_token_ids + q_token_ids[1:])
            # batch_decoder_ids.append([0] + a_token_ids)

            batch_texts.append([context, t_ans, p_ans, s])

        # batch decoder input
        # batch_decoder_ids = sequence_padding(batch_decoder_ids)
        # batch_decoder_ids = torch.tensor(batch_decoder_ids).long()

        # batch decoder mask
        # batch_decoder_mask = compute_attention_bias__gpt(batch_decoder_ids[..., :-1], pad_id=self.tokenizer._token_pad_id)

        # batch labels
        # batch_label_ids = batch_decoder_ids.clone()[..., 1:]
        # label_mask = batch_label_ids == self.tokenizer._token_pad_id
        # batch_label_ids[label_mask] = -100

        # batch encoder input
        batch_encoder_ids, batch_encoder_masks = sequence_padding(batch_encoder_ids, with_mask=True)
        batch_encoder_ids = torch.tensor(batch_encoder_ids).long()
        batch_encoder_masks = torch.tensor(batch_encoder_masks).long()

        return batch_encoder_ids, batch_encoder_masks, batch_texts



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='mt5-large/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='models/')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workspace/FasterTransformer/cache/")
    parser.add_argument("--max_ite", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default='build/lib/libth_t5.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')
    parser.add_argument("--top_k", type=int, default=1, help="top k for sampling")
    parser.add_argument("--top_p", type=float, default=0.0, help="top p for sampling")
    parser.add_argument("--beam_width", type=int, default=3, help="beam width for beam search")

    args = parser.parse_args()
    np.random.seed(0) # rouge score use sampling to compute the score

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0


    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"
    hf_model_location = args.hf_model_location

    tokenizer = SpTokenizer('models/sentencepiece_cn.model', token_start=None, token_end='</s>')


    #  load model
    ckpt_config = configparser.ConfigParser()

    ckpt_config_path = os.path.join(ft_model_location, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    else:
        assert False, "[ERROR] This example only support loading model with FT format directly."

    weight_data_type = np.float32
    weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
    relative_attention_max_distance = 128
    encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                d_model=ckpt_config.getint("encoder", "d_model"),
                                d_kv=ckpt_config.getint("encoder", "d_kv"),
                                d_ff=ckpt_config.getint("encoder", "d_ff"),
                                num_layers=ckpt_config.getint("encoder", "num_layers"),
                                num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                num_heads=ckpt_config.getint("encoder", "num_heads"),
                                relative_attention_num_buckets=ckpt_config.getint(
                                    "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                )
    decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                d_model=ckpt_config.getint("decoder", "d_model"),
                                d_kv=ckpt_config.getint("decoder", "d_kv"),
                                d_ff=ckpt_config.getint("decoder", "d_ff"),
                                num_layers=ckpt_config.getint("decoder", "num_layers"),
                                num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                num_heads=ckpt_config.getint("decoder", "num_heads"),
                                relative_attention_num_buckets=ckpt_config.getint(
                                    "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                )
    assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
    assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj

    t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
    use_gated_activation = encoder_config.is_gated_act
    position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
    activation_type = encoder_config.feed_forward_proj

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
    # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
    tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )

    start_time = datetime.datetime.now()
    ft_encoder_weight.load_from_bin(ft_model_location, "Megatron")
    stop_time = datetime.datetime.now()
    print(f"[INFO] load FT encoder model spend {(stop_time - start_time).total_seconds()} sec")
    start_time = datetime.datetime.now()
    ft_decoding_weight.load_from_bin(ft_model_location, "Megatron")
    stop_time = datetime.datetime.now()
    print(f"[INFO] load FT decoding model spend {(stop_time - start_time).total_seconds()} sec")
    if args.data_type == "fp32":
        ft_encoder_weight.to_float()
        ft_decoding_weight.to_float()
    elif args.data_type == "fp16":
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()
    elif args.data_type == "bf16":
        ft_encoder_weight.to_bfloat16()
        ft_decoding_weight.to_bfloat16()

    ft_encoder_weight.to_cuda()
    ft_decoding_weight.to_cuda()

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    remove_padding = True
    ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets,
                                0, # num_experts
                                [], # moe_layer_index
                                relative_attention_max_distance, False, q_scaling, tensor_para_size,
                                pipeline_para_size, t5_with_bias,
                                position_embedding_type, moe_k=0, activation_type=activation_type)

    ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                decoder_config.vocab_size, q_scaling,
                                decoder_config.relative_attention_num_buckets,
                                0, # num_experts
                                [], # moe_layer_index
                                max_distance=relative_attention_max_distance,
                                tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                moe_k=0, activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

    ft_t5 = FTT5(ft_encoder, ft_decoding)


    dev_examples = read_examples()
    dev_dataset = WordsGenDataset(args, dev_examples, tokenizer=tokenizer)
    dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)

    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=dev_sampler,
        batch_size=16,
        collate_fn=dev_dataset.collate_fn,
        drop_last=True,
        pin_memory=True
    )

    pred_log = []
    em, em2, total = 0, 0, 0
    for k, batch in enumerate(dev_dataloader):
        query_token_ids, query_token_masks, texts = batch
        with torch.no_grad():
            output, ft_output_len, cum_log_probs = ft_t5((query_token_ids, query_token_masks),
                                          None,
                                          args.beam_width,
                                          args.max_seq_len,
                                          top_k=None,
                                          top_p=None,
                                          beam_search_diversity_rate=0.0,
                                          is_return_output_log_probs=False,
                                          len_penalty=1.0,
                                          is_return_cum_log_probs=True)
        if k < 3:
            print('====== pre ======')
            print(output_lines)
            print(cum_log_probs)

        output_lines = [tokenizer.decode([int(idx) for idx in output[0][beam_idx][:ft_output_len[0][beam_idx]]]) for beam_idx in range(args.beam_width)]
        output_lines = ["".join(output_line) for output_line in output_lines]
        cum_log_probs = [str(cum_log_probs[0][beam_idx]) for beam_idx in range(args.beam_width)]

        if k < 3:
            print('====== post ======')
            print(output_lines)
            print(cum_log_probs)

        pred_log.append({
            "msg": texts[0],
            "true": texts[1],
            "pred": texts[2],
            "score": texts[3],
            "ft_res": {
                "preds": output_lines,
                "scores": cum_log_probs
            }
        })

    json.dump(pred_log, Path("res.json").open('w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()