import pathlib
import json
import argparse
from transformers import BartTokenizer
import evaluate
from tqdm import tqdm
import warnings
# from h5_utils import rename_weight
from torch.profiler import profile, record_function, ProfilerActivity

def get_test_data(path):
    with open(path) as f:
        return json.load(f)

def from_pretrained(path, ebno_db, bin_conv_method, channel_type,
                    embedding_dim, num_embeddings,
                    fec_type, cdl_model, 
                    scenario, perfect_csi,
                    channel_num_tx_ant=1, channel_num_rx_ant=1,
                    num_bits_per_symbol=4):
    from models.seq2seq_sc import TFSeq2SeqSCForConditionalGeneration
    import transformers
    
    model = TFSeq2SeqSCForConditionalGeneration.from_pretrained(
        path, ebno_db=ebno_db,
        bin_conv_method=bin_conv_method, channel_type=channel_type,
        fec_type=fec_type, cdl_model=cdl_model,
        scenario=scenario, perfect_csi=perfect_csi,
        channel_num_tx_ant=channel_num_tx_ant, channel_num_rx_ant=channel_num_rx_ant,
        num_bits_per_symbol=num_bits_per_symbol,
        embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    
    return model

def predict(path, ebno_db, 
        tokenizer, batch_size, test_data_path, 
        num_beams, bin_conv_method, channel_type,
        fec_type, cdl_model, 
        scenario, perfect_csi,
        channel_num_tx_ant, channel_num_rx_ant,
        num_bits_per_symbol,
        embedding_dim, num_embeddings):
    import tensorflow as tf
    max_len = 32

    # load model
    model = from_pretrained(path, ebno_db, 
                bin_conv_method=bin_conv_method,
                channel_type=channel_type,
                fec_type=fec_type, cdl_model=cdl_model,
                scenario=scenario, perfect_csi=perfect_csi,
                channel_num_tx_ant=channel_num_tx_ant, channel_num_rx_ant=channel_num_rx_ant,
                num_bits_per_symbol=num_bits_per_symbol,
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings)

    # # load testset
    test_data = get_test_data(test_data_path)
    input_sentences = [d['input'] for d in test_data]
    input_ids = tokenizer(input_sentences, return_tensors="tf", 
                padding='max_length', truncation=True, max_length=max_len).input_ids
    testset = tf.data.Dataset.from_tensor_slices(input_ids)        

    # inference
    pred_sentences = []
    bers=[]
    for input_ids in tqdm(testset.batch(batch_size).prefetch(tf.data.AUTOTUNE)):
        output = model(input_ids) # To get BER
        pred_batch = model.generate(input_ids, max_new_tokens=max_len, num_beams=num_beams,
                                    top_k=4, penalty_alpha=0.6, do_sample=False)
        
        output_strs = tokenizer.batch_decode(pred_batch,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        pred_sentences.extend(output_strs)
        bers.append(output.ber.numpy())

        print(f'output_strs: {output_strs}') # XXX
        print(f'ber: {output.ber.numpy()}') # XXX

    mean_ber = sum(bers) / len(bers)

    res = {
        'input': input_sentences,
        'pred': pred_sentences,
        'refs': [d['refs'] for d in test_data],
        'mean_ber': mean_ber,
    }
    return res

def get_predictions(path, ebno_db, test_data_path, 
        prediction_json_path, batch_size, 
        tokenizer, num_beams, bin_conv_method,
        channel_type, fec_type, cdl_model,
        scenario, perfect_csi,
        channel_num_tx_ant,channel_num_rx_ant, num_bits_per_symbol,
        embedding_dim, num_embeddings):
    path = pathlib.Path(path)
    if not prediction_json_path.exists():
        print('Missing predictions.json')
        
        print('Calculate flops.')
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True,record_shapes=True) as prof:
            with record_function("model_inference"):
                
        
                res = predict(
                    path=path, 
                    ebno_db=ebno_db, 
                    tokenizer=tokenizer, 
                    batch_size=batch_size, 
                    test_data_path=test_data_path, 
                    num_beams=num_beams,
                    bin_conv_method=bin_conv_method,
                    channel_type=channel_type,
                    fec_type=fec_type,
                    cdl_model=cdl_model,
                    scenario=scenario,
                    perfect_csi=perfect_csi,
                    channel_num_tx_ant=channel_num_tx_ant,
                    channel_num_rx_ant=channel_num_rx_ant,
                    num_bits_per_symbol=num_bits_per_symbol,
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings
                )
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # save result
        with open(prediction_json_path, 'w') as f:
            json.dump(res, f, indent=4)
    else:
        with open(prediction_json_path, 'r') as f:
            res = json.load(f)
    return res

def calc_bleu(predictions, tokenizer, multi_ref, **kwargs):
    bleu = evaluate.load('bleu')
    if multi_ref:
        warnings.warn('BLEU does not support multiple references')
    tokenize = lambda l: tokenizer(l, add_special_tokens=False).input_ids
    results = bleu.compute(
        references=predictions['input'],
        predictions=predictions['pred'],               
        tokenizer=tokenize,
        max_order=4)
    return results

def calc_sbert(predictions, batch_size, multi_ref, **kwargs):
    from sentence_transformers import SentenceTransformer, util
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
            model_name_or_path='all-MiniLM-L6-v2',
            device=device)

    sentences1 = predictions['pred']
    print(f'{sentences1=}')
    print(f'{len(sentences1)=}')
    print(f'{multi_ref=}')
    if not multi_ref:
        refs = [[s] for s in predictions['input']]
    else:
        refs = predictions['refs']
    print(f'{refs=}')
    print(f'{len(refs)=}')
    print(f'{len(refs[0])=}')

    def calc_cos_score(model, hyp_embedding, ref_sentences):
        hyp = hyp_embedding.reshape((1, -1))
        refs = model.encode(ref_sentences, convert_to_tensor=True)
        scores = util.cos_sim(hyp, refs)
        scores = scores.reshape((-1)).tolist()
        return {
                'scores': scores,
                'max_score': max(scores),
                'mean_score': sum(scores) / len(scores),
                }
        

    # compute embedding
    pred_embed = model.encode(sentences1, batch_size=batch_size, convert_to_tensor=True)
    print(f'{pred_embed.shape=}')
    N = pred_embed.shape[0]
    scores = [
            calc_cos_score(model, pred_embed[i], refs[i]) for i in range(N)
    ]
    max_scores = [s['max_score'] for s in scores]
    mean_score = sum(max_scores)/len(max_scores)
    return {
        'metric': 'sentence textual similarity',
        'mean_score': mean_score,
        'scores': scores,
    }

METRIC_TO_SCORER = {
        'bleu': calc_bleu,
        'sbert': calc_sbert,
}

def calc(args):
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
    
    path = args.path
    metric = args.metric
    batch_size = args.batch_size

    # rename weight name in tf_model.h5 for BinConv
    # rename_weight(path, RENAME_MAP)

    # VQ-VAE arguments
    if args.bin_conv_method=='vector_quantization':
        assert (args.embedding_dim is not None and args.num_embeddings is not None), 'Set embedding_dim and num_embeddings.'

    predictions = get_predictions(
            path, 
        ebno_db=args.ebno_db,
        prediction_json_path=args.prediction_json_path,
        test_data_path=args.testset_path, 
        batch_size=batch_size, 
        tokenizer=tokenizer,
        num_beams=args.num_beams,
        bin_conv_method=args.bin_conv_method,
        channel_type=args.channel_type,
        fec_type=args.fec_type,
        cdl_model=args.cdl_model,
        scenario=args.scenario,
        perfect_csi=args.perfect_csi,
        channel_num_tx_ant=args.channel_num_tx_ant,
        channel_num_rx_ant=args.channel_num_rx_ant,
        embedding_dim=int(args.embedding_dim),
        num_embeddings=int(args.num_embeddings),
        num_bits_per_symbol=args.num_bits_per_symbol)
    scorer = METRIC_TO_SCORER[metric]
    results = scorer(
        predictions=predictions,
        tokenizer=tokenizer,
        batch_size=batch_size,
        multi_ref=args.multi_ref,
    )

    # Add mean_ber
    results['mean_ber'] = predictions['mean_ber']
    
    # dump result
    with open(args.result_json_path, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', metavar='checkpoint_path', type=pathlib.Path)
    parser.add_argument('-m', '--metric', choices = list(METRIC_TO_SCORER.keys()), dest='metric')
    parser.add_argument('-b', '--batch-size', default=4, type=int, dest='batch_size')
    parser.add_argument('-e', '--ebno-db', required=True, type=float, dest='ebno_db')
    parser.add_argument('--testset-path', 
            required=True, type=pathlib.Path, dest='testset_path')
    parser.add_argument('--prediction-json-path', 
            required=True, 
            type=pathlib.Path,
            dest='prediction_json_path',
            help='Required. Output path of prediction result cache json file. \
                  If the file exists, the prediction result will be reused')
    parser.add_argument('--result-json-path', 
            default=pathlib.Path('./result.json'),
            type= pathlib.Path,
            dest='result_json_path')
    parser.add_argument('--tokenizer', 
            default='facebook/bart-base', 
            dest='tokenizer')
    parser.add_argument('--num-beams', 
            default=1, 
            type=int,
            dest='num_beams')
    parser.add_argument('--multi-ref', 
            action='store_true', 
            dest='multi_ref')
    parser.add_argument('--bin-conv-method', default='tanh')
    parser.add_argument('--channel-type', default='AWGN')
    parser.add_argument('--fec-type', default='Polar5G')
    parser.add_argument('--fec-num-iter', default=6)
    parser.add_argument('--cdl-model', default='A') # CDL
    parser.add_argument('--scenario', default='umi') # 3GPP-38.901
    parser.add_argument('--perfect-csi', default=True) # 3GPP-38.901
    parser.add_argument('--channel-num-tx-ant', default=2)
    parser.add_argument('--channel-num-rx-ant', default=2)
    parser.add_argument('--num-bits-per-symbol', default=4)
    parser.add_argument('--embedding-dim', default=2) # vector quantization
    parser.add_argument('--num-embeddings', default=1024) # vector quantization
    parser.add_argument('--calc-flops', default=False, type=bool)
    args = parser.parse_args()
    print(f'{args=}')

    calc(args)

if __name__ == '__main__':
    main()
