import torch
import transformers
from torch import nn
from loguru import logger

import nncomp.registry as R


class TransformerDecoderBase(torch.nn.Module):

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
        return outputs

    def _reorder_cache(self, past, beam_idx):
        return self.transformer._reorder_cache(past, beam_idx)


@R.ModuleRegistry.add
class AutoTransformerDecoder(TransformerDecoderBase):

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        encoder_num_features: int,
        use_triangle_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = transformers.AutoConfig.for_model(
            model_name,
            vocab_size=vocab_size,
            output_hidden_states=True,
            **kwargs,
        )
        self.transformer = transformers.AutoModelForCausalLM.from_config(
            self.config,
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
class AutoPreTrainedTransformerDecoder(TransformerDecoderBase):

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        encoder_num_features: int,
        reset_word_embeddings: bool = False,
        use_triangle_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True,
            **kwargs,
        )
        self.transformer = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
        )
        self.use_triangle_attention_mask = use_triangle_attention_mask
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
