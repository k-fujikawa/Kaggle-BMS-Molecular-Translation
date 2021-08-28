from typing import Type

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions  # NOQA

import nncomp.registry as R


RNN_MODULES = dict(
    lstm=nn.LSTM,
    gru=nn.GRU,
)
RNN_NUM_STATES = dict(
    lstm=2,
    gru=1,
)


class AttnRNNEncoder(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        encoder_num_features: int,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 12,
        layernorm: bool = True,
        rnn_module: str = "lstm",
    ):
        """
        RNN Decoder with Attention.

        Parameters
        ----------
        in_dim: int
            RNNへの入力次元数
        encoder_num_features: int
            画像側のchannel数
        hidden_size: int
            中間層の次元数
        num_hidden_layers: int
            レイヤー数
        p_dropout: float
            Dropoutの確率
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.rnns = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.state_initializers = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.num_states = RNN_NUM_STATES[rnn_module]
        for i in range(num_hidden_layers):
            self.state_initializers.append(nn.ModuleList([
                nn.Linear(encoder_num_features, hidden_size)
                for _ in range(self.num_states)
            ]))
            self.attentions.append(
                nn.MultiheadAttention(
                    embed_dim=input_size,
                    num_heads=num_attention_heads,
                    kdim=encoder_num_features,
                    vdim=encoder_num_features,
                )
            )
            self.rnns.append(
                RNN_MODULES[rnn_module](
                    input_size=input_size * 2,
                    hidden_size=hidden_size,
                )
            )
            input_size = hidden_size
            if layernorm:
                self.layernorms.append(nn.LayerNorm(hidden_size))

    def init_hidden_states(self, x=None):
        if self.num_states > 1:
            return [
                [initializer(x)[None] for initializer in initializers]
                for initializers in self.state_initializers
            ]
        else:
            return [
                initializers[0](x)[None]
                for initializers in self.state_initializers
            ]

    def _reorder_cache(self, past, beam_idx):
        return [
            [past_state_type[:, beam_idx] for past_state_type in past_layer]
            for past_layer in past
        ]

    def forward(
        self,
        h_tokens,
        encoder_hidden_states,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is None:
            past_key_values = self.init_hidden_states(
                encoder_hidden_states.mean(dim=1)
            )
        # RNN / Attentionが期待するshapeに調整: (seqlen, batch, hidden)
        h_tokens = h_tokens.permute(1, 0, 2)
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        hs, attn_output_weights, rnn_states = [], [], []
        for i in range(len(self.rnns)):
            attn_output, _attn_output_weights = self.attentions[i](
                query=h_tokens,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
            )
            rnn_input = torch.cat([h_tokens, attn_output], dim=-1)
            h_tokens, _rnn_states = self.rnns[i](
                rnn_input,
                past_key_values[i]
            )
            if len(self.layernorms) > 0:
                h_tokens = self.layernorms[i](h_tokens)
            hs.append(h_tokens)
            attn_output_weights.append(_attn_output_weights)
            rnn_states.append(_rnn_states)

        # (seqlen, batch, hidden) -> (batch, seqlen, hidden)
        hidden_states = torch.stack(hs).permute(0, 2, 1, 3)

        return BaseModelOutputWithPastAndCrossAttentions(
            past_key_values=rnn_states,
            hidden_states=hidden_states,
        )


@R.ModuleRegistry.add
class AttnRNNDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        encoder_num_features: int,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 12,
        p_dropout: float = 0.,
        concat_hs: bool = True,
        layernorm: bool = True,
        rnn_module: str = "lstm",
    ):
        """
        RNN Decoder with Attention.

        Parameters
        ----------
        vocab_size: int
            入出力のInChI語彙数
        embed_dim: int
            Embedding次元数
        encoder_num_features: int
            画像側のchannel数
        hidden_size: int
            中間層の次元数
        num_hidden_layers: int
            レイヤー数
        p_dropout: float
            Dropoutの確率
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = AttnRNNEncoder(
            input_size=embed_size,
            encoder_num_features=encoder_num_features,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            layernorm=layernorm,
            rnn_module=rnn_module,
        )
        self.dropout = nn.Dropout(p_dropout)
        self.concat_hs = concat_hs
        if self.concat_hs:
            hidden_dim = hidden_size * num_hidden_layers
        else:
            hidden_dim = hidden_size
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _reorder_cache(self, past, beam_idx):
        return self.encoder._reorder_cache(past, beam_idx)

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        past_key_values=None,
        **kwargs,
    ):
        batch_size, seqlen = input_ids.shape
        h_tokens = self.embedding(input_ids)
        outputs = self.encoder(
            h_tokens=h_tokens,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )
        # (seqlen, batch, hidden) -> (batch, seqlen, hidden)
        if self.concat_hs:
            h = outputs.hidden_states.permute(1, 2, 3, 0)\
                .reshape(batch_size, seqlen, -1)
        else:
            h = outputs.hidden_states[-1]
        y = self.output(self.dropout(h))
        return {
            "logits": y,
            "past_key_values": outputs.past_key_values,
        }
