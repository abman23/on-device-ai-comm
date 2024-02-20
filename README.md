<!-- # seq2seq-SC

## Citation

```bash
@misc{lee2022seq2seqSC,
    author = {Lee, Ju-Hyung and Lee, Dong-Ho and Sheen, Eunsoo and Choi, Thomas and Pujara, Jay and Kim, Joongheon},
    title = {Seq2Seq-SC: End-to-End Semantic Communication Systems with Pre-trained Language Model},
    journal={arXiv preprint arXiv:2210.15237},
    year = {2022},
}
``` -->

# On-device AI Communication

This project presents an innovative on-device AI communication system, integrating pre-trained language models with physical layer communications. This repo is the implementation for our paper ["Integrating Pre-Trained Language Model with Physical Layer Communications"](https://arxiv.org/abs/2402.11656).

## Highlights
- Integration with Physical Layer Communications: We seamlessly integrate language models with physical communication layers, optimizing for noisy environments through a novel noise-tuning method.
- Efficiency and Robustness: Our approach reduces transmission size by 50% without compromising message integrity, demonstrating superior performance under standard 3GPP channel models.
- Pre-trained Models for Generalization: Utilizing pre-trained BART models, we enhance the system's ability to generalize across different data domains, making it highly adaptable.

## Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/abman23/on-device-ai-comm.git
cd on-device-ai-comm
conda env create -f environment.yml
conda activate on-device-ai-comm
```

## Data Preprocessing

### Europarl dataset

```bash
data_path=data/europarl
mkdir -p $data_path
cd $data_path
wget -P /tmp http://www.statmt.org/europarl/v7/europarl.tgz
tar zxf /tmp/europarl.tgz

europarl_dataset="$data_path/txt/en"
out_dir="$data_path/processed"
njobs=4

mkdir -p $out_dir
python -m preprocess.europarl -j $njobs -o $out_dir $europarl_dataset
```

<!-- ### AllNLI

Run `./scripts/preprocess_allnli.sh` or the following commands

```bash
data_path=data/allnli
mkdir -p $data_path
wget -P $data_path https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/AllNLI.jsonl.gz
gunzip $data_path/AllNLI.jsonl.gz

allnli_dataset="$data_path/AllNLI.jsonl"
out_dir="$data_path/processed"

mkdir -p $out_dir
python -m preprocess.allnli -o $out_dir $allnli_dataset
``` -->

### Flickr30K 

To download the dataset, go to [Flickr30K](http://hockenmaier.cs.illinois.edu/DenotationGraph/) and fill out the form to get the downloadable link. 

```bash
data_path="data/flickr"
dataset_path="${data_path}/flickr30k.tar.gz"
out_dir="$data_path/processed"

mkdir -p $out_dir

tar xzf ${dataset_path} -C $data_path
python -m preprocess.flickr30k \
    -o "$out_dir/flickr30k.json" \
    "${data_path}/results_20130124.token"
```

## Train

You can run `scripts/train.sh`. Otherwise, you can train by running the follwing commands. Below is an example for training on-device ai communication system over CDL-A 5 ~ 15dB with vector quantizer.

```bash
output_dir='checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15'
trainset_path='data/europarl/processed/train.csv'
devset_path='data/europarl/processed/test.csv'

mkdir -p $output_dir

python train.py \
    --model_name_or_path facebook/bart-base \
    --config_name facebook/bart-base \
    --tokenizer_name facebook/bart-base \
    --train_file "$trainset_path" \
    --validation_file "$devset_path" \
    --test_file "$devset_path" \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size  4 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --no_use_fast_tokenizer \
    --num_beams 1 \
    --pad_to_max_length \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --output_dir $output_dir \
    --ebno_db_min 5 \
    --ebno_db_max 15 \
    --channel_type "CDL" \
    --fec_type "Polar5G" \
    --fec_num_iter 20 \
    --cdl_model "A" \
    --channel_num_tx_ant "2" \
    --channel_num_rx_ant "2" \
    --num_bits_per_symbol "4" \
    --bin_conv_method "vector_quantization" \
    --embedding_dim 2 \
    --num_embeddings 1024 \
    --dataset_config 3.0.0
```

## Evaluation

You can use the script `scripts/eval.sh` or the following commands:

```bash
# BLEU score
eval_ebno_db="4"
metric="bleu" # bleu, sbert
testset_path='data/flickr/processed/flickr30k.json'

checkpoint_dir='checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15'
output_dir='checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15/CDL-A'

mkdir -p $output_dir

fec_type="Polar5G" # Polar5G, LDPC5G
fec_num_iter=20
channel_num_tx_ant="2"
channel_num_rx_ant="2"
num_bits_per_symbol="4"
EVAL_NUM_BEAMS="1"

python eval.py \
    -m "${metric}" \
    -b 8 \
    -e "${eval_ebno_db}" \
    --result-json-path "${output_dir}/flickr_${metric}_${eval_ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --prediction-json-path "${output_dir}/flickr_prediction_${eval_ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --fec-type "${fec_type}" \
    --fec-num-iter "${fec_num_iter}" \
    --channel-type "CDL" \
    --cdl-model "A" \
    --channel-num-tx-ant "${channel_num_tx_ant}" \
    --channel-num-rx-ant "${channel_num_rx_ant}" \
    --num-bits-per-symbol "${num_bits_per_symbol}" \
    --bin-conv-method "vector_quantization" \
    --embedding-dim 2 \
    --num-embeddings 1024 \
    --num-beams "${EVAL_NUM_BEAMS}" \
    --testset-path "${testset_path}" \
    $checkpoint_dir
```


