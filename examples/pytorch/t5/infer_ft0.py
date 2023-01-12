# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import argparse
import json
import numpy as np
import os
from pathlib import Path
import sys
import time
import torch
import torch.distributed as dist
from datasets import load_dataset, load_metric
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config, MT5ForConditionalGeneration, MT5Config
from bojone_snippets import *
from bojone_tokenizers import *

from tqdm import tqdm
import configparser
import math
from collections import namedtuple
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/FasterTransformer")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder

# 模型配置
max_passage_len = 256
max_question_len = 128

batch_size = 8

def load_data():
    dev_D = []
    for i, line in enumerate(Path("/data/app/words_generation" + "/" + "selected_message.txt").open()):
        msg = line.strip()[1:-1].replace('""', '"')[:256]
        
        # 预处理 去表情
        emoji_tokens = re.findall(r'\[\w+\]', msg, re.I|re.M)
        for token in emoji_tokens:
            msg = msg.replace(token, '')

        dev_D.append([msg, '没有'])

        if i == batch_size * 2 - 1:
            break

    print(f'data loaded ! dev data size: {len(dev_D)}')
    return dev_D

def compute_attention_bias__gpt(input_ids, pad_id=0):
    """
    [b,max_len,max_len]

    """
    bs, max_len = input_ids.size()
    att_mask = torch.zeros((bs, max_len, max_len))

    seq_lens = (input_ids != pad_id).sum(-1)  # [b,1]

    for i, l in enumerate(seq_lens):
        att_mask[i][:l, :l] = torch.tril(torch.ones((l, l)))

    return att_mask

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """
        batch_decoder_ids, batch_encoder_ids = [], []
        batch_texts = []
        for is_end, (p, q) in self.sample(random):
            p_token_ids, _ = self.kwargs["tokenizer"].encode(p, maxlen=max_passage_len)
            a_token_ids, _ = self.kwargs["tokenizer"].encode('这句的搜索词条是什么？', maxlen=max_passage_len)
            q_token_ids, _ = self.kwargs["tokenizer"].encode(q, maxlen=max_question_len)

            batch_encoder_ids.append(p_token_ids + a_token_ids[1:])
            batch_decoder_ids.append([0] + q_token_ids)

            batch_texts.append([p, q])

            if len(batch_decoder_ids) == self.batch_size or is_end:
                # decoder input
                batch_decoder_ids = sequence_padding(batch_decoder_ids)
                batch_decoder_ids = torch.tensor(batch_decoder_ids).long()

                # decoder mask
                batch_decoder_mask = compute_attention_bias__gpt(batch_decoder_ids[..., :-1],
                                                                 pad_id=self.kwargs["tokenizer"]._token_pad_id)

                # labels
                batch_label_ids = batch_decoder_ids.clone()[..., 1:]
                label_mask = batch_label_ids == self.kwargs["tokenizer"]._token_pad_id
                batch_label_ids[label_mask] = -100

                # encoder input
                batch_encoder_ids, batch_encoder_masks = sequence_padding(batch_encoder_ids, with_mask=True)
                batch_encoder_ids = torch.tensor(batch_encoder_ids).long()
                batch_encoder_masks = torch.tensor(batch_encoder_masks).long()

                yield batch_encoder_ids, batch_decoder_ids[..., :-1], \
                      batch_encoder_masks, batch_decoder_mask, batch_label_ids, batch_texts

                batch_decoder_ids, batch_encoder_ids = [], []
                batch_texts = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/T5/HF/t5-base/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='/models/T5/HF/t5-base/')
    parser.add_argument('--disable_summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_ft', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default='./lib/libth_t5.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')

    args = parser.parse_args()

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0

    disable_summarize = args.disable_summarize
    test_hf = args.test_hf
    test_ft = args.test_ft

    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"
    hf_model_location = args.hf_model_location
    config_path = os.path.join(hf_model_location, 'config.json')
    checkpoint_path = os.path.join(hf_model_location, 'pytorch_model.bin') # 大语料预训练用
    dict_path = os.path.join(hf_model_location, "sentencepiece_cn.model")
    state_dict = os.path.join(hf_model_location, "words_model_neg_em14.pt")

    # 加载词典
    tokenizer = SpTokenizer(str(dict_path), token_start=None, token_end='</s>')
    # dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    if rank == 0 and test_hf:
        start_time = datetime.datetime.now()
        # 加载模型
        config = MT5Config.from_pretrained(config_path)
        if args.data_type == "fp32":
            model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, config=config, torch_dtype=torch.float32).cuda()
        elif args.data_type == "fp16":
            model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, config=config, torch_dtype=torch.float16).cuda()
        elif args.data_type == "bf16":
            model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, config=config, torch_dtype=torch.bfloat16).cuda()
        model.load_state_dict(torch.load(state_dict))
        stop_time = datetime.datetime.now()
        print(f"[INFO] load HF model spend {(stop_time - start_time).total_seconds()} sec")

    if test_ft:
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

    if disable_summarize:
        top_k = 1
        output_len = args.max_seq_len
    else:
        top_k = 1
        output_len = args.max_seq_len

    dev_data = load_data()
    dev_generator = data_generator(dev_data, batch_size=batch_size, tokenizer=tokenizer)

    skip_count = 10
    total_count = 500
    total_time = 0
    samples_count = 0
    pred_error_count = 0
    output = Path("output.txt").open('w')
    ft_output = Path("ft_output.txt").open('w')
    ft_input_ids = namedtuple("ft_input_ids", ['input_ids', 'attention_mask'])
    for step, batch in enumerate(dev_generator):
        texts = batch[-1]
        cur_batch = [_.cuda() for _ in batch[:-1]]
        query_token_ids = cur_batch[0]
        with torch.no_grad():
            time_s = time.time()
            torch.cuda.synchronize()
            sequence_outputs = model.generate(input_ids=query_token_ids,
                                        # do_sample=True,
                                        # top_p=0.95,
                                        num_beams=3,
                                        # num_beam_groups=1,
                                        max_length=max_question_len,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer._token_pad_id,
                                        eos_token_id=config.eos_token_id,
                                        decoder_start_token_id=config.decoder_start_token_id,
                                        output_scores=True,
                                        return_dict_in_generate=True
                                    )

            ft_input_token = ft_input_ids(input_ids=cur_batch[0], attention_mask=cur_batch[2])
            ft_decoding_outputs, ft_output_len, ft_output_log_probs, ft_cum_log_probs = ft_t5(input_token=ft_input_token,
                                          inputs_embeds=None,
                                          beam_size=1,
                                          max_seq_len=128,
                                          top_k=top_k,
                                          top_p=0.0,
                                          beam_search_diversity_rate=0.0,
                                          len_penalty=1.0,
                                          is_return_output_log_probs=True,
                                          is_return_cum_log_probs=True)
            torch.cuda.synchronize()
            time_e = time.time()
            print(f"step:{step}, ft inference time={(time_e - time_s)*1000} ms")
        if skip_count <= step and step < total_count:
            total_time += (time_e - time_s)
        # if step >= total_count:
        #     break
        # print(ft_decoding_outputs.shape, ft_cum_log_probs.shape)
        pred_seqs, pred_seq_scores = sequence_outputs.sequences.cpu().numpy(), sequence_outputs.sequences_scores.cpu().numpy()
        print(pred_seq_scores)
        print(np.squeeze(ft_cum_log_probs))
        print(np.squeeze(ft_output_len))
        print(np.squeeze(ft_cum_log_probs) / pred_seq_scores)
        hf_pre_q = []
        ft_pre_q = []
        for pred_q, pred_s, passage in zip(pred_seqs, pred_seq_scores, texts):
            pred_q = ''.join(tokenizer.decode([int(idx) for idx in pred_q]))
            hf_pre_q.append(pred_q)
            json_output = json.dumps(
                {
                    "msg": passage,
                    "pred": pred_q,
                    "score": pred_s.tolist()
                }, ensure_ascii=False)

            output.write(json_output + "\n")
        # print(ft_cum_log_probs.shape, ft_cum_log_probs)
        for pred_q, pred_s, passage in zip(ft_decoding_outputs, ft_cum_log_probs, texts):
            pred_result = []
            for pred_ in pred_q:
                pred_result.append(''.join(tokenizer.decode([int(idx) for idx in pred_])))
            ft_pre_q.append(pred_result[0])
            json_output = json.dumps(
                {
                    "msg": passage,
                    "pred": pred_result,
                    "score": pred_s.tolist()
                }, ensure_ascii=False)

            ft_output.write(json_output + "\n")
        assert len(hf_pre_q) == len(ft_pre_q)
        assert len(hf_pre_q) == len(texts)
        samples_count += len(hf_pre_q)
        for pred_q, ft_pred_q_0 in zip(hf_pre_q, ft_pre_q):
            if pred_q != ft_pred_q_0:
                pred_error_count += 1
    print(f"error_match: {pred_error_count}/{samples_count} = {pred_error_count/samples_count}")
    print(f"torch avg inference time={total_time * 1000 / (total_count - skip_count)} ms")
    return
    if not disable_summarize:
        datapoint = dataset_cnn['test'][0]
        if test_ft:
            summary_ft, _ = summarize_ft(datapoint)
            if rank == 0:
                print('---------------------------------------------------------')
                print('FT Generated : ')
                print(' Article : ', datapoint['article'])
                print('\n Highlights : ', datapoint['highlights'])
                print('\n Summary : ', summary_ft)
                print('---------------------------------------------------------')
                metric_ft.add_batch(predictions=[summary_ft], references=[datapoint['highlights']])

        if test_hf and rank == 0:
            summary_hf, _ = summarize_hf(datapoint)
            print('---------------------------------------------------------')
            print('HF Generated : ')
            print(' Article : ', datapoint['article'])
            print('\n Highlights : ', datapoint['highlights'])
            print('\n Summary : ', summary_hf)
            print('---------------------------------------------------------')
            metric_hf.add_batch(predictions=[summary_hf], references=[datapoint['highlights']])

    ft_time = 0.0
    hf_time = 0.0
    for data_point_idx in tqdm(range(1, 11490, int(11490 / args.max_ite))):
        try:
            datapoint = dataset_cnn['test'][data_point_idx]

            start_time = datetime.datetime.now()
            if test_ft:
                summary_ft, tokens_ft = summarize_ft(datapoint)
            stop_time = datetime.datetime.now()
            ft_time += (stop_time - start_time).total_seconds()

            if rank == 0 and ((test_hf and not disable_summarize) or disable_summarize):
                start_time = datetime.datetime.now()
                summary_hf, tokens_hf = summarize_hf(datapoint)
                stop_time = datetime.datetime.now()
                hf_time += (stop_time - start_time).total_seconds()

            if rank == 0:
                if not disable_summarize:
                    if test_ft:
                        metric_ft.add_batch(predictions=[summary_ft], references=[datapoint['highlights']])
                    if test_hf:
                        metric_hf.add_batch(predictions=[summary_hf], references=[datapoint['highlights']])
                else:
                    tokens.append((tokens_ft, tokens_hf))
        except:
            print('Error with datapoint : ', data_point_idx)

    def compute_exact_match(tokens, n_tokens=[1, 10, 25, 50, 100, 150, 200, 250]):
        em_metrics = []
        for t in n_tokens:
            errors = 0
            total = 0
            for ft_tokens, hf_tokens in tokens:
                if len(ft_tokens) > t and len(hf_tokens) > t:
                    total = total + 1
                    if not np.array_equal(ft_tokens[:t], hf_tokens[:t]):
                        errors = errors + 1

            if total > 0:
                print(f"{t}-token exact match acc: {100*(1-errors/total):.2f}")
                em_metrics.append(1 - errors / total)
            else:
                em_metrics.append(np.nan)

        return em_metrics

    if rank == 0:
        if not disable_summarize:
            if test_ft:
                computed_metrics_ft = metric_ft.compute()

            if test_hf:
                computed_metrics_hf = metric_hf.compute()

                print(f'Hugging Face (total latency: {hf_time} sec)')
                for key in computed_metrics_hf.keys():
                    print(f'{key} : {computed_metrics_hf[key].mid[2]*100}')

            if test_ft:
                print(f'Faster Transformers (total latency: {ft_time} sec)')
                for key in computed_metrics_ft.keys():
                    print(f'{key} : {computed_metrics_ft[key].mid[2]*100}')
                if args.rougeLsum_threshold != None:
                    assert computed_metrics_ft["rougeLsum"].mid[2] * \
                        100 >= args.rougeLsum_threshold, "[INFO] TEST FAIL !"
                    print(f"[INFO] TEST PASS !")
        else:
            em_metrics = compute_exact_match(tokens)


if __name__ == '__main__':
    main()
