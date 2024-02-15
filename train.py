#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

# This file is based on 
# https://github.com/huggingface/transformers/blob/main/examples/tensorflow/summarization/run_summarization.py
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import nltk 
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import matplotlib.pyplot as plt

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    KerasMetricCallback,
    TFTrainingArguments,
    set_seed,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode
from transformers.optimization_tf import create_optimizer
from train.args import ModelArguments, DataTrainingArguments, summarization_name_mapping, Seq2SeqSCArguments
from models.seq2seq_sc import TFSeq2SeqSCForConditionalGeneration # XXX

logger = logging.getLogger(__name__)

from datetime import datetime

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments, Seq2SeqSCArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, seq2seq_sc_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, seq2seq_sc_args = parser.parse_args_into_dataclasses()

    if training_args.fp16:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    current_time = str(datetime.now())
    sys.stdout = open(f'./{training_args.output_dir}/train_stdout_{current_time}.log','a')
    sys.stderr = open(f'./{training_args.output_dir}/train_stderr_{current_time}.log','a')
    sys.stdout.write("\n\n-----"+current_time+"-----\n\n")
    sys.stderr.write("\n\n-----"+current_time+"-----\n\n")

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # region Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # endregion

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # region Load datasets
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # endregion

    # region Load model config and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    # endregion

    # region Dataset preprocessing
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, and/or `do_eval`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        assert prefix is not None
        for i, inp in enumerate(inputs):
            if inp is None:
                print(i, inputs[i], inputs[i-1], inputs[i+1])
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    else:
        train_dataset = None

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    else:
        eval_dataset = None
    # endregion

    # region Text preprocessing
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


    # endregion
    with training_args.strategy.scope():
        # region Prepare model
        if model_args.model_name_or_path == '' or model_args.model_name_or_path is None:
            # Create a BART model with random initialization
            model = TFSeq2SeqSCForConditionalGeneration(
                ebno_db=seq2seq_sc_args.ebno_db,
                ebno_db_min=seq2seq_sc_args.ebno_db_min,
                ebno_db_max=seq2seq_sc_args.ebno_db_max,
                fec_k=seq2seq_sc_args.k,
                fec_n=seq2seq_sc_args.n,
                num_bits_per_symbol=seq2seq_sc_args.num_bits_per_symbol,
                channel_type=seq2seq_sc_args.channel_type,
                cdl_model=seq2seq_sc_args.cdl_model,
                channel_num_tx_ant=seq2seq_sc_args.channel_num_tx_ant,
                channel_num_rx_ant=seq2seq_sc_args.channel_num_rx_ant,
                fec_type=seq2seq_sc_args.fec_type,
                bin_conv_method=seq2seq_sc_args.bin_conv_method,
                embedding_dim=seq2seq_sc_args.embedding_dim,
                num_embeddings=seq2seq_sc_args.num_embeddings,
                do_train=training_args.do_train,
                config=config,
            )
            print(f'Random Initialized without pre-trained weights.')
        else:
            model_cls = TFSeq2SeqSCForConditionalGeneration
            # https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                ebno_db=seq2seq_sc_args.ebno_db,
                ebno_db_min=seq2seq_sc_args.ebno_db_min,
                ebno_db_max=seq2seq_sc_args.ebno_db_max,
                fec_k=seq2seq_sc_args.k,
                fec_n=seq2seq_sc_args.n,
                num_bits_per_symbol=seq2seq_sc_args.num_bits_per_symbol,
                channel_type=seq2seq_sc_args.channel_type,
                cdl_model=seq2seq_sc_args.cdl_model,
                channel_num_tx_ant=seq2seq_sc_args.channel_num_tx_ant,
                channel_num_rx_ant=seq2seq_sc_args.channel_num_rx_ant,
                fec_type=seq2seq_sc_args.fec_type,
                bin_conv_method=seq2seq_sc_args.bin_conv_method,
                embedding_dim=seq2seq_sc_args.embedding_dim,
                num_embeddings=seq2seq_sc_args.num_embeddings,
                do_train=training_args.do_train,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            print(f'Loaded {model_args.model_name_or_path}')

        model.resize_token_embeddings(len(tokenizer))

        # endregion

        # region Prepare TF Dataset objects
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=128,  # Reduce the number of unique shapes for XLA, especially for generation
            return_tensors="tf",
        )

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas

        # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
        # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
        # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
        # yourself if you use this method, whereas they are automatically inferred from the model input names when
        # using model.prepare_tf_dataset()
        # For more info see the docs:
        # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
        # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset

        tf_train_dataset = model.prepare_tf_dataset(
            train_dataset,
            collate_fn=data_collator,
            batch_size=total_train_batch_size,
            shuffle=True,
        ).with_options(dataset_options)
        tf_eval_dataset = model.prepare_tf_dataset(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=total_eval_batch_size,
            shuffle=False,
        ).with_options(dataset_options)
        # endregion

        # region Optimizer, loss and LR scheduling
        num_train_steps = int(len(tf_train_dataset) * training_args.num_train_epochs)
        if training_args.warmup_steps > 0:
            num_warmup_steps = training_args.warmup_steps
        elif training_args.warmup_ratio > 0:
            num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
        else:
            num_warmup_steps = 0
        if training_args.do_train:
            optimizer, lr_schedule = create_optimizer(
                init_lr=training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
                weight_decay_rate=training_args.weight_decay,
                adam_global_clipnorm=training_args.max_grad_norm,
            )
        else:
            optimizer = None

        # endregion

        # region Metric and KerasMetricCallback
        if training_args.do_eval:
            metric = evaluate.load("rouge")

            if data_args.val_max_target_length is None:
                data_args.val_max_target_length = data_args.max_target_length

            gen_kwargs = {
                "max_length": data_args.val_max_target_length if data_args is not None else config.max_length,
                "num_beams": data_args.num_beams,
                "no_repeat_ngram_size": 0,  # Not supported under XLA right now, and some models set it by default
            }

            def compute_metrics(preds):
                predictions, labels = preds
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metrics = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                # Only print the mid f-measures, but there are a lot of other statistics in there too!
                metrics = {key: round(val * 100, 4) for key, val in metrics.items()}
                return metrics

            # The KerasMetricCallback allows metrics that are too complex to write as standard Keras metrics
            # to be computed each epoch. Any Python code can be included in the metric_fn. This is especially
            # useful for metrics like BLEU and ROUGE that perform string comparisons on decoded model outputs.
            # For more information, see the docs at
            # https://huggingface.co/docs/transformers/main_classes/keras_callbacks#transformers.KerasMetricCallback

            metric_callback = KerasMetricCallback(
                metric_fn=compute_metrics,
                eval_dataset=tf_eval_dataset,
                predict_with_generate=True,
                use_xla_generation=True,
                generate_kwargs=gen_kwargs,
            )
            callbacks = [metric_callback]

            # Define a callback to save the model
            checkpoint_callback = ModelCheckpoint(
                filepath= training_args.output_dir + '/model_epoch_{epoch:02d}_loss_{loss:.4f}.h5',
                monitor='loss',  # metric name
                save_best_only=True,  # Save only if the monitored quantity improves
                save_weights_only=True,  # Only save the model weights
                verbose=1  # Print more information
            )

            # Add the checkpoint callback to list of callbacks
            callbacks.append(checkpoint_callback)
        else:
            callbacks = []
        # endregion

        # region Training
        model.compile(optimizer=optimizer, jit_compile=training_args.xla)
        eval_metrics = None
        if training_args.do_train:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size = {total_train_batch_size}")
            logger.info(f"  Total optimization steps = {num_train_steps}")

            if training_args.xla and not data_args.pad_to_max_length:
                logger.warning(
                    "XLA training may be slow at first when --pad_to_max_length is not set "
                    "until all possible shapes have been compiled."
                )
            history = model.fit(tf_train_dataset, epochs=int(training_args.num_train_epochs), callbacks=callbacks)
            # e.g., When epoch3, history.history={'loss': [3.3186378479003906, 1.8291417360305786, 1.5430701971054077], 'rouge1': [49.4591, 57.3391, 66.4159], 'rouge2': [36.3391, 50.6601, 60.5605], 'rougeL': [49.1986, 58.1554, 65.7636], 'rougeLsum': [49.112, 58.124, 65.7636]}
            eval_metrics = {key: val[-1] for key, val in history.history.items()}
        # endregion

        # region Validation

        if training_args.do_eval and not training_args.do_train:
            # Do a standalone evaluation run
            logger.info("Evaluation...")

            # Compiling generation with XLA yields enormous speedups, see https://huggingface.co/blog/tf-xla-generate
            @tf.function(jit_compile=True)
            def generate(**kwargs):
                return model.generate(**kwargs)

            for batch, labels in tf_eval_dataset:
                batch.update(gen_kwargs)
                generated_tokens = generate(**batch)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            eval_metrics = metric.compute(use_stemmer=True)

            result = {key: round(val * 100, 4) for key, val in eval_metrics.items()}
            logger.info(result)
        # endregion

        if training_args.output_dir is not None and eval_metrics is not None:
            output_eval_file = os.path.join(training_args.output_dir, "all_results.json")
            with open(output_eval_file, "w") as writer:
                writer.write(json.dumps(eval_metrics))

            if history is not None:
                ### --- Draw plot --- ###
                loss = history.history['loss']
                rouge1 = history.history['rouge1']
                rouge2 = history.history['rouge2']
                rougeL = history.history['rougeL']
                rougeLsum = history.history['rougeLsum']

                # Create an array of epoch numbers
                epochs = np.arange(1, len(loss) + 1)

                # Plotting the validation loss
                fig, ax1 = plt.subplots(figsize=(10, 6))
                color = 'tab:blue'
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss', color=color)
                ax1.plot(epochs, loss, label='Loss', color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                # Set y-axis range for validation loss
                ax1.set_ylim([0, max(loss)+0.2])

                # Creating a secondary y-axis for ROUGE scores
                ax2 = ax1.twinx()

                color = 'tab:green'
                ax2.set_ylabel('ROUGE Scores', color=color)
                ax2.plot(epochs, rouge1, label='ROUGE-1', color=color)
                ax2.plot(epochs, rouge2, label='ROUGE-2', color='tab:red')
                ax2.plot(epochs, rougeL, label='ROUGE-L', color='tab:purple')
                ax2.plot(epochs, rougeLsum, label='ROUGE-Lsum', color='tab:orange')
                ax2.tick_params(axis='y', labelcolor=color)
                # Set y-axis range for ROUGE scores
                ax2.set_ylim([min(rouge1) - 10, 100])

                # Add legend and grid
                fig.tight_layout()
                fig.legend(loc='lower right')
                plt.grid(True)

                # Save the plot
                output_plot_file = os.path.join(training_args.output_dir, "validation_loss_rouge_scores.png")
                plt.savefig(output_plot_file)


        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()