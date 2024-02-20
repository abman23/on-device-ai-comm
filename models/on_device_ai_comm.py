'''
MIMO OFDM Transmissions over the CDL Channel Model
https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html
'''
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from transformers import TFBartPretrainedModel, TFBartForConditionalGeneration
from transformers.models.bart.modeling_tf_bart import TFBartMainLayer, BartConfig, shift_tokens_right, TFBartEncoder
from transformers.modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqModelOutput
from transformers.modeling_tf_utils import unpack_inputs, TFModelInputType
import tensorflow as tf
import numpy as np

import sionna
sionna.Config.xla_compat=True

from transformers.utils import (
    logging,
)

from .channels import ChannelAWGN, ChannelCDL, ChannelSL, ChannelFlatFading
from .utils import tensor_to_binary_v2, binary_to_tensor_v2, replace_nan, get_ber
from .vq_vae import VectorQuantizer

from transformers.tf_utils import shape_list
from transformers.modeling_tf_outputs import TFSeq2SeqLMOutput, TFBaseModelOutput

logger = logging.get_logger("transformers")

@dataclass
class TFEncoderChannelModelOutput(TFBaseModelOutput):
    """Output of TFAISrcEncoderAndChannel that includes
        - AI-Src encoder
        - channel encoder
        - mapper
        - channel
        - demapper
        - channel decoder
    """
    ber: Optional[tf.Tensor] = None

@dataclass
class TFOnDeviceAICMainLayerOutput(TFSeq2SeqModelOutput):
    """Output of TFOnDeviceAICMainLayer"""
    ber: Optional[tf.Tensor] = None

@dataclass
class TFOnDeviceAICOutput(TFSeq2SeqLMOutput):
    """Output of TFOnDeviceAICForConditionalGeneration"""
    ber: Optional[tf.Tensor] = None

class TFAISrcEncoderAndChannel(tf.keras.layers.Layer):
    """
    This class includes AI-Src-Encoder and Channel(Channel En/Decoder, Channel, Mapper, etc.)
    """

    def __init__(self, 
            ai_src_encoder: TFBartEncoder, 
            vq_layer: VectorQuantizer,
            ebno_db,
            ebno_db_min,
            ebno_db_max,
            channel_type,
            cdl_model,
            scenario,
            perfect_csi,
            fec_type,
            channel_num_tx_ant, 
            channel_num_rx_ant,
            num_bits_per_symbol=4,
            fec_k=512,
            fec_n=1024,
            fec_num_iter=6,
            bin_conv_method='tanh',
            do_train=False
            ):
        # NOTE: setting layer name as follows seems strange, 
        # but it allows HuggingFace to load pretrained weight properly
        super().__init__(name='model/model/')
        self.config = ai_src_encoder.config
        self.ai_src_encoder = ai_src_encoder
        self.ai_src_encoder.trainable = False

        # If Training TFOnDeviceAICForConditionalGeneration or not
        self.do_train = do_train
        
        # make sure data types are proper.
        num_bits_per_symbol = int(num_bits_per_symbol)

        # Configure Channel Model
        if channel_type == 'AWGN':
            ch_config = {
                'fec_type': fec_type,
                'num_bits_per_symbol': num_bits_per_symbol,
                'fec_n': fec_n,
                'fec_k': fec_k,
                'ebno_db' : ebno_db,
                'ebno_db_min': ebno_db_min,
                'ebno_db_max': ebno_db_max,
                'fec_num_iter': fec_num_iter,
            }
        elif channel_type == 'CDL':
            ch_config = {
                'fec_type': fec_type,
                'cdl_model': cdl_model,
                'channel_num_tx_ant': channel_num_tx_ant,
                'channel_num_rx_ant': channel_num_rx_ant,
                'num_bits_per_symbol': num_bits_per_symbol,
                'ebno_db' : ebno_db,
                'ebno_db_min': ebno_db_min,
                'ebno_db_max': ebno_db_max,
                'fec_num_iter': fec_num_iter,
            }
        elif channel_type == '3GPP-38.901':
            ch_config = {
                'fec_type': fec_type,
                'scenario': scenario,
                'perfect_csi': perfect_csi,
                'channel_num_tx_ant': channel_num_tx_ant,
                'channel_num_rx_ant': channel_num_rx_ant,
                'num_bits_per_symbol': num_bits_per_symbol,
                'ebno_db' : ebno_db,
                'ebno_db_min': ebno_db_min,
                'ebno_db_max': ebno_db_max,
                'fec_num_iter': fec_num_iter,
            }
        elif channel_type == 'FlatFading':
            ch_config = {
                'fec_type': fec_type,
                'channel_num_tx_ant': channel_num_tx_ant,
                'channel_num_rx_ant': channel_num_rx_ant,
                'num_bits_per_symbol': num_bits_per_symbol,
                'fec_n': fec_n,
                'fec_k': fec_k,
                'ebno_db' : ebno_db,
                'ebno_db_min': ebno_db_min,
                'ebno_db_max': ebno_db_max,
                'fec_num_iter': fec_num_iter,
            }
        else:
            raise ValueError('Invalid Channel type. Channel type should be AWGN or CDL')
        
        # define vq
        if bin_conv_method == 'vector_quantization':
            self.vq_layer =vq_layer
            self.num_embeddings = vq_layer.num_embeddings
            self.embedding_dim = vq_layer.embedding_dim

        # setup
        self._setup_channel(channel_type, ch_config)
        self._setup_bin_conv(bin_conv_method)
        
    def _setup_channel(self, channel_type, ch_config):
        if channel_type == 'AWGN':
            self.channel_model = ChannelAWGN(
                fec_type=ch_config['fec_type'], 
                num_bits_per_symbol=ch_config['num_bits_per_symbol'],
                fec_n=ch_config['fec_n'], 
                fec_k=ch_config['fec_k'],
                ebno_db=ch_config['ebno_db'],
                ebno_db_min=ch_config['ebno_db_min'],
                ebno_db_max=ch_config['ebno_db_max'],
                fec_num_iter=ch_config['fec_num_iter']
            )
        elif channel_type == 'CDL':
            self.channel_model = ChannelCDL(
                fec_type=ch_config['fec_type'],
                cdl_model=ch_config['cdl_model'],
                channel_num_tx_ant=ch_config['channel_num_tx_ant'],
                channel_num_rx_ant=ch_config['channel_num_rx_ant'],
                num_bits_per_symbol=ch_config['num_bits_per_symbol'],
                ebno_db=ch_config['ebno_db'],
                ebno_db_min=ch_config['ebno_db_min'],
                ebno_db_max=ch_config['ebno_db_max'],
                fec_num_iter=ch_config['fec_num_iter']
            )
        elif channel_type == '3GPP-38.901':
            self.channel_model = ChannelSL(
                fec_type=ch_config['fec_type'],
                scenario=ch_config['scenario'],
                perfect_csi=ch_config['perfect_csi'],
                channel_num_tx_ant=ch_config['channel_num_tx_ant'],
                channel_num_rx_ant=ch_config['channel_num_rx_ant'],
                num_bits_per_symbol=ch_config['num_bits_per_symbol'],
                ebno_db=ch_config['ebno_db'],
                ebno_db_min=ch_config['ebno_db_min'],
                ebno_db_max=ch_config['ebno_db_max'],
                fec_num_iter=ch_config['fec_num_iter']
            )
        elif channel_type == 'FlatFading':
            self.channel_model = ChannelFlatFading(
                fec_type=ch_config['fec_type'], 
                channel_num_tx_ant=ch_config['channel_num_tx_ant'],
                channel_num_rx_ant=ch_config['channel_num_rx_ant'],
                num_bits_per_symbol=ch_config['num_bits_per_symbol'],
                fec_n=ch_config['fec_n'], 
                fec_k=ch_config['fec_k'],
                ebno_db=ch_config['ebno_db'],
                ebno_db_min=ch_config['ebno_db_min'],
                ebno_db_max=ch_config['ebno_db_max'],
                fec_num_iter=ch_config['fec_num_iter']
            )
        else:
            raise ValueError('Invalid Channel type. Channel type should be AWGN or CDL')
    
    def _setup_bin_conv(self, bin_conv_method):
        self.bin_conv_method = bin_conv_method
        logger.info(f'{bin_conv_method=}')
        if bin_conv_method == 'naive':
            self._tensor_to_binary = [tensor_to_binary_v2]
            self._binary_to_tensor = [binary_to_tensor_v2]
        elif bin_conv_method == 'tanh':
            self._tensor_to_binary = [tensor_to_binary_v2]
            self._binary_to_tensor = [binary_to_tensor_v2, tf.math.tanh]
        elif bin_conv_method == 'vector_quantization':
            self._tensor_to_binary = [self.vq_layer.get_code_indices, tensor_to_binary_v2]
            self._binary_to_tensor = [binary_to_tensor_v2, 
                                      self.vq_layer.handle_invalid_values, 
                                      self.vq_layer.reconstruct_with_indices]
        else:
            raise ValueError(f'Invalid bin_conv_method: {bin_conv_method}')


    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # run AI-Src encoder(BART)
        ai_src_encoder_outputs = self.ai_src_encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        shape = tf.shape(ai_src_encoder_outputs.last_hidden_state)
        ai_src_encoder_output = ai_src_encoder_outputs.last_hidden_state
        
        # binarize (and get codebook indices if vector quantizing)
        for f in self._tensor_to_binary:
            ai_src_encoder_output = f(ai_src_encoder_output)
        ai_src_encoder_output_binary = ai_src_encoder_output
        
        # add channel noise to binarized ai_src_encoder output
        last_hidden_state_binary = self.channel_model(ai_src_encoder_output_binary)
        
        # convert to tensor and denoise (and reconstruct tensors(to feed semantic decoder) using codebook if vector quantizing)
        last_hidden_state = last_hidden_state_binary
        for f in self._binary_to_tensor:
            last_hidden_state = f(last_hidden_state)
        last_hidden_state_pred = tf.reshape(last_hidden_state, shape)

        last_hidden_state_pred = replace_nan(last_hidden_state_pred, 0) # convert all NaN values to zero
        if (self.bin_conv_method != 'naive'):
            tf.debugging.assert_all_finite(last_hidden_state_pred, 'should not have nan/inf/-inf')

        # calculate BER if eval.
        if not self.do_train:
            last_hidden_state_binary = tf.reshape(last_hidden_state_binary, tf.shape(ai_src_encoder_output_binary))
            ber = get_ber(ai_src_encoder_output_binary, last_hidden_state_binary)
        else:
            # While training, does not calculate BER.
            ber = tf.constant(-1.0, dtype=tf.float32)
        
        return TFEncoderChannelModelOutput(
            last_hidden_state=last_hidden_state_pred,
            hidden_states=ai_src_encoder_outputs.hidden_states,
            attentions=ai_src_encoder_outputs.attentions,
            ber=ber,
        )


class TFOnDeviceAICMainLayer(tf.keras.layers.Layer):

    def __init__(self,
                 config: BartConfig,
                 bart_main_layer: TFBartMainLayer,
                 ebno_db,
                 ebno_db_min,
                 ebno_db_max,
                 fec_k=512,
                 fec_n=1024,
                 fec_num_iter=6,
                 num_bits_per_symbol=4,
                 channel_type = 'AWGN',
                 cdl_model='A',
                 scenario='umi',
                 perfect_csi=True,
                 channel_num_tx_ant=1, 
                 channel_num_rx_ant=1,
                 fec_type=None,
                 bin_conv_method='tanh',
                 embedding_dim=2,
                 num_embeddings=1024,
                 do_train=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.shared = bart_main_layer.get_input_embeddings()
        self.shared.trainable = False

        self.bin_conv_method = bin_conv_method
        # VectorQuantizer layer
        if self.bin_conv_method == 'vector_quantization':
            print(f'{embedding_dim=}')
            print(f'{num_embeddings=}') # number of codebooks.
            assert (embedding_dim is not None) and (num_embeddings is not None) \
                , "For vector_quantization, set embedding_dim and num_embeddings arguments."
            self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, name="vector_quantizer")
        else:
            self.vq_layer = None

        # ai source encoder
        self.encoder = TFAISrcEncoderAndChannel(
            ai_src_encoder=bart_main_layer.encoder,
            vq_layer = self.vq_layer,
            ebno_db=ebno_db,
            ebno_db_min=ebno_db_min,
            ebno_db_max=ebno_db_max,
            fec_k=fec_k,
            fec_n=fec_n,
            fec_num_iter=fec_num_iter,
            num_bits_per_symbol=num_bits_per_symbol,
            channel_type=channel_type,
            cdl_model=cdl_model,
            scenario=scenario,
            perfect_csi=perfect_csi,
            channel_num_tx_ant=channel_num_tx_ant,
            channel_num_rx_ant=channel_num_rx_ant,
            fec_type=fec_type,
            bin_conv_method=bin_conv_method,
            do_train=do_train)
        self.decoder = bart_main_layer.decoder

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared    

    @unpack_inputs
    def call(self,
             input_ids: Optional[TFModelInputType] = None,
             attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_attention_mask: Optional[Union[np.ndarray,
                                                    tf.Tensor]] = None,
             decoder_position_ids: Optional[Union[np.ndarray,
                                                  tf.Tensor]] = None,
             head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             cross_attn_head_mask: Optional[Union[np.ndarray,
                                                  tf.Tensor]] = None,
             encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
             past_key_values: Optional[Tuple[Tuple[Union[np.ndarray,
                                                         tf.Tensor]]]] = None,
             inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
             decoder_inputs_embeds: Optional[Union[np.ndarray,
                                                   tf.Tensor]] = None,
             use_cache: Optional[bool] = None,
             output_attentions: Optional[bool] = None,
             output_hidden_states: Optional[bool] = None,
             return_dict: Optional[bool] = None,
             training: Optional[bool] = False,
             **kwargs) -> Union[TFOnDeviceAICMainLayerOutput, Tuple[tf.Tensor]]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id,
                self.config.decoder_start_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a TFBaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs,
                                            TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None,
            )

        # If the user passed a TFBaseModelOutput for encoder_outputs, we wrap it in a tuple when return_dict=False
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # call VectorQuantizer to train VectorQuantizer
        if self.bin_conv_method == 'vector_quantization':
            # encoder_outputs.last_hidden_state = self.vq_layer(encoder_outputs.last_hidden_state)
            self.vq_layer(encoder_outputs.last_hidden_state)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return TFOnDeviceAICMainLayerOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ber=encoder_outputs.ber,
        )
    

class TFOnDeviceAICForConditionalGeneration(TFBartForConditionalGeneration):
    def __init__(self,
                 config,
                 ebno_db=None,
                 ebno_db_min=None,
                 ebno_db_max=None,
                 fec_k=512,
                 fec_n=1024,
                 fec_num_iter=6,
                 num_bits_per_symbol=4,
                 channel_type = 'AWGN',
                 cdl_model='A',
                 scenario='umi',
                 perfect_csi=True,
                 channel_num_tx_ant = 1,
                 channel_num_rx_ant = 1,
                 fec_type = None,
                 bin_conv_method='tanh', 
                 embedding_dim=None,
                 num_embeddings=None,
                 do_train=False,
                 *inputs,
                 **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFOnDeviceAICMainLayer(
            config,
            bart_main_layer=self.model,
            ebno_db=ebno_db,
            ebno_db_min=ebno_db_min,
            ebno_db_max=ebno_db_max,
            fec_k=fec_k,
            fec_n=fec_n,
            fec_num_iter=fec_num_iter,
            num_bits_per_symbol=num_bits_per_symbol,
            channel_type=channel_type,
            cdl_model=cdl_model,
            scenario=scenario,
            perfect_csi=perfect_csi,
            channel_num_tx_ant=channel_num_tx_ant,
            channel_num_rx_ant=channel_num_rx_ant,
            fec_type=fec_type,
            bin_conv_method=bin_conv_method,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            do_train=do_train,
            name="model")

        self.bin_conv_method = bin_conv_method
        self.VQ_LOSS_WEIGHT = 0.01

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        cross_attn_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # add weighted vq loss into masked_lm_loss
        if self.bin_conv_method == 'vector_quantization' and masked_lm_loss is not None:
            vq_loss = self.VQ_LOSS_WEIGHT * sum(self.model.vq_layer.losses)
            # tf.print(
            #     masked_lm_loss, 
            #     vq_loss, 
            #     sep=',',
            #     output_stream='./joohan/seq2seq-sc2/loss.log')
            masked_lm_loss += vq_loss # multiply weight to vq_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return TFOnDeviceAICOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
            ber=outputs.ber,
        )
