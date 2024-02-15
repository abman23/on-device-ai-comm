#!/usr/bin/env python
from slurm_helper import submit_slurm, get_attr_from_name
from pathlib import Path
import sys
import os

# Common
fec_type='LDPC5G' # Polar5G, LDPC5G
fec_num_iter=6
channel_num_tx_ant="2"
channel_num_rx_ant="2"
num_bits_per_symbol="4"

# Eval
eval_ebno_dbs = [1,2,3,4,5]
eval_cdl_models = ['A']
EVAL_NUM_BEAMS="1"
calc_flops = False

# Train
train_cdl_model = 'A'
# SINGLE_EBNO_DB=10
EBNO_DB_MIN=5
EBNO_DB_MAX=15
NUM_EPOCH = 3
embedding_dim=2
num_embeddings=1024
TEST_DATASET={
    'flickr': 'data/flickr/processed/flickr30k.json',
    'europarl': 'data/europarl/processed/test.json',
    'europarl_train_set': 'data/europarl/processed/train.json', # Only for check if train was processed.
    'europarl_train_set_small': 'data/europarl/processed/train_small.json' # Only for check if train was processed.
}
TEST_DATASET_TYPE='europarl' # flickr, europarl

# usage: eval_rl.py [-h] [-m {bleu,sbert}] [-b BATCH_SIZE] -e EBNO_DB --testset-path TESTSET_PATH --prediction-json-path
#                   PREDICTION_JSON_PATH [--result-json-path RESULT_JSON_PATH] [--tokenizer TOKENIZER]
#                   [--num-beams NUM_BEAMS] [--multi-ref]
#                   checkpoint_path

def train():
    # random (w/o pre-training)
    # output_dir = f'checkpoints/seq2seq-europarl-sc/ebnodb_{EBNO_DB_MIN}_{EBNO_DB_MAX}_random_ep{NUM_EPOCH}_{fec_type}_cdl{train_cdl_model}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}_naive'
    # output_dir = f'checkpoints/seq2seq-europarl-sc/ebnodb_{EBNO_DB_MIN}_{EBNO_DB_MAX}_random_ep{NUM_EPOCH}_{fec_type}_{fec_num_iter}iter_cdl{train_cdl_model}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}_VQ_{embedding_dim}_{num_embeddings}'
    # non-random
    output_dir = f'checkpoints/seq2seq-europarl-sc/ebnodb_{EBNO_DB_MIN}_{EBNO_DB_MAX}_ep{NUM_EPOCH}_{fec_type}_{fec_num_iter}iter_cdl{train_cdl_model}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}_naive_wo_nan'
    # output_dir = f'checkpoints/seq2seq-europarl-sc/ebnodb_{EBNO_DB_MIN}_{EBNO_DB_MAX}_ep{NUM_EPOCH}_{fec_type}_{fec_num_iter}iter_cdl{train_cdl_model}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}_tanh'
    # output_dir = f'checkpoints/seq2seq-europarl-sc/ebnodb_{EBNO_DB_MIN}_{EBNO_DB_MAX}_ep{NUM_EPOCH}_{fec_type}_{fec_num_iter}iter_cdl{train_cdl_model}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}_VQ_{embedding_dim}_{num_embeddings}'
    create_dir(output_dir)
    cmds = []
    if not (Path(output_dir) / 'tf_model.h5').exists():
        train_cmd = ['python', 
            'train.py',
            '--model_name_or_path', 'facebook/bart-base', # remove this param if you want to train with random initialization
            '--config_name', 'facebook/bart-base',
            '--tokenizer_name', 'facebook/bart-base',
            '--train_file', 'data/europarl/processed/train.csv',
            '--validation_file', 'data/europarl/processed/test.csv',
            '--test_file', 'data/europarl/processed/test.csv',
            '--preprocessing_num_workers', 4,
            "--per_device_train_batch_size", 3,
            "--per_device_eval_batch_size",  3,
            "--num_train_epochs", NUM_EPOCH,
            "--do_train",
            "--do_eval",
            "--save_total_limit", 1,
            "--no_use_fast_tokenizer",
            "--num_beams", 1,
            '--pad_to_max_length',
            '--overwrite_output_dir',
        
            # bart
            "--max_source_length", 64,
            "--max_target_length", 64,
            '--output_dir', output_dir,

            # model config
            # '--ebno_db', SINGLE_EBNO_DB,
            '--ebno_db_min', EBNO_DB_MIN,
            '--ebno_db_max', EBNO_DB_MAX,
            '--dataset_config', '3.0.0',
            '--channel_type', 'CDL',
            '--fec_type', fec_type,
            '--fec_num_iter', fec_num_iter,
            '--cdl_model', train_cdl_model,
            '--channel_num_tx_ant', channel_num_tx_ant,
            '--channel_num_rx_ant', channel_num_rx_ant,
            '--num_bits_per_symbol', num_bits_per_symbol,
            '--bin_conv_method', 'naive', # naive, tanh, vector_quantization
            # '--embedding_dim', embedding_dim,
            # '--num_embeddings', num_embeddings
            ]
        train_cmd = list(map(str, train_cmd))
        cmds.append(train_cmd)
        submit_slurm(
            cmds, 
            output_dir=output_dir, 
            prefix='train_eval_',
            gpu='v100')

def eval():
    for eval_ebno_db in eval_ebno_dbs:
        for cdl_model in eval_cdl_models:
            checkpoint_dir="checkpoints/seq2seq-europarl-sc/ebnodb_5_15_ep3_LDPC5G_6iter_cdlA_2_2_4_tanh"
            output_dir = f'checkpoints/seq2seq-europarl-sc/ebnodb_5_15_ep3_LDPC5G_6iter_cdlA_2_2_4_tanh/CDL-{cdl_model}/{TEST_DATASET_TYPE}'

            create_dir(output_dir)

            cmds = []
            
            eval_cmd = ['python', 
                'eval.py',
                '-m', 'bleu',
                '-b', '16',
                '-e', f'{eval_ebno_db}',
                '--result-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_bleu_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--prediction-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_prediction_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--fec-type', fec_type,
                '--fec-num-iter', fec_num_iter,
                '--channel-type', 'CDL',
                '--cdl-model', cdl_model,
                '--channel-num-tx-ant', channel_num_tx_ant,
                '--channel-num-rx-ant', channel_num_rx_ant,
                '--num-bits-per-symbol', num_bits_per_symbol,
                '--num-beams', EVAL_NUM_BEAMS,
                '--bin-conv-method', 'tanh', # tanh, naive
                '--testset-path', TEST_DATASET[TEST_DATASET_TYPE],
                '--calc-flops', calc_flops,
                checkpoint_dir]
            eval_vq_cmd = ['python', 
                'eval.py',
                '-m', 'bleu',
                '-b', '16',
                '-e', f'{eval_ebno_db}',
                '--result-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_bleu_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--prediction-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_prediction_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--fec-type', fec_type,
                '--fec-num-iter', fec_num_iter,
                '--channel-type', 'CDL',
                '--cdl-model', cdl_model,
                '--channel-num-tx-ant', channel_num_tx_ant,
                '--channel-num-rx-ant', channel_num_rx_ant,
                '--num-bits-per-symbol', num_bits_per_symbol,
                '--bin-conv-method', 'vector_quantization',
                '--embedding-dim', 2,
                '--num-embeddings', 1024,
                '--num-beams', EVAL_NUM_BEAMS,
                '--testset-path', TEST_DATASET[TEST_DATASET_TYPE],
                '--calc-flops', calc_flops,
                checkpoint_dir]
            eval_cmd = list(map(str, eval_cmd))
            eval_vq_cmd = list(map(str, eval_vq_cmd))
            cmds.append(eval_cmd)
            # cmds.append(eval_vq_cmd)

            eval_cmd = ['python', 
                'eval.py',
                '-m', 'sbert',
                '-b', '16',
                '-e', f'{eval_ebno_db}',
                '--result-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_sbert_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--prediction-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_prediction_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--fec-type', fec_type,
                '--fec-num-iter', fec_num_iter,
                '--channel-type', 'CDL',
                '--cdl-model', cdl_model,
                '--channel-num-tx-ant', channel_num_tx_ant,
                '--channel-num-rx-ant', channel_num_rx_ant,
                '--num-bits-per-symbol', num_bits_per_symbol,
                '--num-beams', EVAL_NUM_BEAMS,
                '--bin-conv-method', 'tanh', # tanh, naive
                '--testset-path', TEST_DATASET[TEST_DATASET_TYPE],
                '--calc-flops', calc_flops,
                checkpoint_dir]
            eval_vq_cmd = ['python', 
                'eval.py',
                '-m', 'sbert',
                '-b', '16',
                '-e', f'{eval_ebno_db}',
                '--result-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_sbert_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--prediction-json-path', f'{output_dir}/{TEST_DATASET_TYPE}_prediction_{eval_ebno_db}dB_{fec_type}_{channel_num_tx_ant}_{channel_num_rx_ant}_{num_bits_per_symbol}.json',
                '--fec-type', fec_type,
                '--fec-num-iter', fec_num_iter,
                '--channel-type', 'CDL',
                '--cdl-model', cdl_model,
                '--channel-num-tx-ant', channel_num_tx_ant,
                '--channel-num-rx-ant', channel_num_rx_ant,
                '--num-bits-per-symbol', num_bits_per_symbol,
                '--bin-conv-method', 'vector_quantization',
                '--embedding-dim', 2,
                '--num-embeddings', 1024,
                '--num-beams', EVAL_NUM_BEAMS,
                '--testset-path', TEST_DATASET[TEST_DATASET_TYPE],
                '--calc-flops', calc_flops,
                checkpoint_dir]
            eval_cmd = list(map(str, eval_cmd))
            eval_vq_cmd = list(map(str, eval_vq_cmd))
            cmds.append(eval_cmd)
            # cmds.append(eval_vq_cmd)
            submit_slurm(
                    cmds, 
                    output_dir=output_dir, 
                    prefix='train_eval_',
                    gpu='v100')

def create_dir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")

if __name__ == "__main__":
    job_type = sys.argv[1] # train, eval
    if job_type == 'train':
        train()
    elif job_type == 'eval':
        eval()
    else:
        raise('Invalid job type. It should be train or eval.')
