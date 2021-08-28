from typing import List

import torch
import transformers
from torch import nn
from loguru import logger
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)

import nncomp.registry as R
from nncomp_molecule.decoders.rnn import AttnRNNEncoder


class TransformerDecoderBaseV2(torch.nn.Module):

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache=False,
        past_key_values=None,
    ):
        batch_size, seq_length = input_ids.shape
        if self.add_encoder_adaptor:
            encoder_hidden_states = self.adaptor(encoder_hidden_states)
        if self.use_triangle_attention_mask and attention_mask is not None:
            attention_mask = torch.tril(
                torch.matmul(
                    attention_mask[:, :, None].float(),
                    attention_mask[:, None, :].float(),
                )
            )
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            output_attentions=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        h = torch.cat([
            outputs.hidden_states[i]
            for i in self.concat_hidden_layers
        ], dim=-1)
        logits = self.output(h)
        outputs = CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        return outputs

    def _reorder_cache(self, past, beam_idx):
        return self.transformer._reorder_cache(past, beam_idx)


@R.ModuleRegistry.add
class AutoTransformerDecoderV2(TransformerDecoderBaseV2):

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        encoder_num_features: int,
        concat_hidden_layers: List[int] = [-1, -2],
        **kwargs,
    ):
        super().__init__()
        self.config = transformers.AutoConfig.for_model(
            model_name,
            vocab_size=vocab_size,
            output_hidden_states=True,
            **kwargs,
        )
        self.transformer = transformers.AutoModel.from_config(
            self.config,
        )
        self.concat_hidden_layers = concat_hidden_layers
        dim = len(concat_hidden_layers) * self.config.hidden_size
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, vocab_size),
        )
        self.use_triangle_attention_mask = use_triangle_attention_mask
        self.add_encoder_adaptor = False
        if self.transformer.config.hidden_size != encoder_num_features:
            self.add_encoder_adaptor = True
            self.adaptor = nn.Linear(
                encoder_num_features,
                self.transformer.config.hidden_size,
            )


@R.ModuleRegistry.add
class AutoPreTrainedTransformerDecoderV2(TransformerDecoderBaseV2):

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        encoder_num_features: int,
        reset_word_embeddings: bool = False,
        use_triangle_attention_mask: bool = True,
        extract_decoder: bool = False,
        concat_hidden_layers: List[int] = [-1, -2],
        **kwargs,
    ):
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True,
            **kwargs,
        )
        self.transformer = transformers.AutoModel.from_pretrained(
            model_name,
            config=self.config,
        )
        if extract_decoder:
            self.transformer = self.transformer.decoder
        self.transformer.resize_token_embeddings(vocab_size)
        if reset_word_embeddings:
            logger.info("Reset word embeddings")
            self.transformer.bert.embeddings.word_embeddings.reset_parameters()
        self.add_encoder_adaptor = False
        if self.transformer.config.hidden_size != encoder_num_features:
            self.add_encoder_adaptor = True
            self.adaptor = nn.Linear(
                encoder_num_features,
                self.transformer.config.hidden_size,
            )
        self.use_triangle_attention_mask = use_triangle_attention_mask
        self.concat_hidden_layers = concat_hidden_layers
        dim = len(concat_hidden_layers) * self.config.hidden_size
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, vocab_size),
        )
        self.add_encoder_adaptor = False
        if self.transformer.config.hidden_size != encoder_num_features:
            self.add_encoder_adaptor = True
            self.adaptor = nn.Linear(
                encoder_num_features,
                self.transformer.config.hidden_size,
            )


@R.ModuleRegistry.add
class AutoTransformerRNNDecoder(nn.Module):

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        encoder_num_features: int,
        concat_hidden_layers: List[int] = [-1, -2],
        rnn_module: str = "lstm",
        **kwargs,
    ):
        super().__init__()
        self.config = transformers.AutoConfig.for_model(
            model_name,
            vocab_size=vocab_size,
            output_hidden_states=True,
            **kwargs,
        )
        self.transformer = transformers.AutoModel.from_config(
            self.config,
        )
        self.rnn = AttnRNNEncoder(
            rnn_module=rnn_module,
            input_dim=self.config.hidden_size,
            image_hidden_dim=self.config.hidden_size,
            token_hidden_dim=self.config.hidden_size,
            n_layers=1,
            n_heads=self.config.num_attention_heads,
            layernorm=True,
        )
        self.concat_hidden_layers = concat_hidden_layers
        dim = len(concat_hidden_layers) * self.config.hidden_size
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, vocab_size),
        )
        self.add_encoder_adaptor = False
        if self.transformer.config.hidden_size != encoder_num_features:
            self.add_encoder_adaptor = True
            self.adaptor = nn.Linear(
                encoder_num_features,
                self.config.hidden_size,
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache=False,
        past_key_values=None,
    ):
        if past_key_values is None:
            past_key_values = dict()
        if self.add_encoder_adaptor:
            encoder_hidden_states = self.adaptor(encoder_hidden_states)
        if attention_mask is not None:
            attention_mask = torch.tril(
                torch.matmul(
                    attention_mask[:, :, None].float(),
                    attention_mask[:, None, :].float(),
                )
            )
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            output_attentions=True,
            use_cache=use_cache,
            past_key_values=past_key_values.get("transformer"),
        )
        rnn_outputs = self.rnn(
            h_tokens=transformer_outputs.hidden_states[-1],
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values.get("rnn"),
        )
        outputs = BaseModelOutputWithPastAndCrossAttentions(
            hidden_states=[
                *transformer_outputs.hidden_states,
                *rnn_outputs.hidden_states,
            ],
            past_key_values=dict(
                transformer=transformer_outputs.past_key_values,
                rnn=rnn_outputs.past_key_values,
            ),
        )
        h = torch.cat([
            outputs.hidden_states[i]
            for i in self.concat_hidden_layers
        ], dim=-1)
        logits = self.output(h)
        outputs = CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        return outputs

    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError()
        return dict(
            transformer=self.transformer._reorder_cache(
                past["transformer"], beam_idx
            ),
            rnn=self.rnn._reorder_cache(
                past["rnn"], beam_idx
            ),
        )
