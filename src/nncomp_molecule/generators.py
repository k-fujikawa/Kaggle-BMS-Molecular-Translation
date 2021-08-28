from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

import torch
import torch.nn.functional as F
from transformers.generation_utils import GenerationMixin
from transformers.file_utils import ModelOutput

from nncomp.preprocessors import SequenceCollateFunction
from nncomp_molecule.preprocessors import (
    normalize_inchi_batch,
    disable_rdlogger,
)


@dataclass
class GenerationConfig:
    is_encoder_decoder: bool = False
    num_beams: int = 8
    num_beam_groups: int = 1
    num_return_sequences: int = 8
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_length: int = 300
    min_length: int = 10
    do_sample: bool = False
    output_scores: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    return_dict_in_generate: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_no_repeat_ngram_size: int = 0
    bad_words_ids: Optional[List[List[int]]] = None
    diversity_penalty: float = 0.0
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: int = 2
    remove_invalid_values: bool = True
    early_stopping: bool = True
    use_cache: bool = True


@dataclass
class TokenPredictionOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[List[torch.Tensor]]] = None


class GeneratedInChIScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            ignore_index=0,
            reduction="none",
        )

    def forward(self, logits, next_token_ids):
        _, _, vocab_size = logits.shape
        loss = self.cross_entropy(
            logits.reshape(-1, vocab_size),
            next_token_ids.reshape(-1),
        )
        loss = loss.reshape(*next_token_ids.shape).sum(dim=1)
        cross_entropy = loss / (next_token_ids != 0).sum(dim=1)
        return cross_entropy.tolist()


class EnsenmbleBeamSearchGenerator(torch.nn.Module, GenerationMixin):
    def __init__(
        self,
        config: GenerationConfig,
        tokenizer: Callable,
        models: List[torch.nn.Module],
        weights: List[float],
        transforms: List[Callable] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.models = torch.nn.ModuleList(models)
        self.weights = weights
        self.transforms = transforms

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        attention_mask: torch.LongTensor = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: List[Dict[str, torch.Tensor]],
        past: List[Any] = None,
        **kwargs,
    ):
        return dict(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            past=past,
        )

    def encode(self, images, n_repeats=None):
        batch_size = len(images)
        if n_repeats is None:
            n_repeats = self.config.num_beams
        if self.transforms is not None:
            images = [
                torch.stack([
                    transform(image=image)["image"]
                    for image in images
                ]).to(self.device)
                for transform in self.transforms
            ]
        else:
            images = [images] * len(self.models)
        encoder_outputs = [
            _model.encode(_images)
            for _images, _model in zip(images, self.models)
        ]
        for i, _encoder_outputs in enumerate(encoder_outputs):
            for k, v in _encoder_outputs.items():
                if v is not None:
                    encoder_outputs[i][k] = v[:, None]\
                        .expand(batch_size, n_repeats, *v.shape[1:])\
                        .reshape(batch_size * n_repeats, *v.shape[1:])
        return encoder_outputs

    def _reorder_cache(
        self,
        past: list,  # (n_models, n_layers, (hs, cs), (seqlen=1, batch, dim))
        beam_idx: torch.Tensor,  # (num_beams * batch)
    ):
        past = [
            model.decoder._reorder_cache(_past, beam_idx)
            for _past, model in zip(past, self.models)
        ]
        return past

    def __call__(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: List[Dict[str, torch.Tensor]],
        attention_mask: torch.Tensor = None,
        past: List[Any] = None,
        use_cache: bool = True,
        **kwargs
    ):
        if past is None:
            past = [None] * len(encoder_outputs)
        if use_cache:
            input_ids = input_ids[:, -1:]
        outputs = [
            model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                **_encoder_outputs,
            )
            for model, _encoder_outputs, past_key_values
            in zip(self.models, encoder_outputs, past)
        ]
        logits = sum([
            w * o["logits"]
            for w, o in zip(self.weights, outputs)
        ]) / sum(self.weights)
        past = [o.get("past_key_values") for o in outputs]
        return TokenPredictionOutput(
            logits=logits,
            past_key_values=past,
        )

    def postprocess(self, outputs: TokenPredictionOutput):
        # Log likelihood の計算
        scores = torch.stack(outputs.scores, dim=1)  # B, シーケンス長, 語彙数
        eos_indexes = (outputs.sequences == self.config.eos_token_id).max(dim=1).indices
        eos_indexes[eos_indexes == 0] = scores.shape[1]  # eos_indexes == 0 のときはEOSトークンが出力シーケンスにない場合
        probs = F.softmax(scores, dim=1)
        probs_max = probs.max(dim=2).values
        skip_initial_length = len([1, 39, 12])
        outputs["log_likelihood"] = [
            p[skip_initial_length:e+1].mean().item()
            for p, e in zip(probs_max, eos_indexes)
        ]

        # token ids -> texts
        outputs["InChI"] = self.tokenizer.decode_batch(
            outputs.sequences.tolist(),
        )

        # normalize InChI
        disable_rdlogger()
        outputs["normed_InChI"] = normalize_inchi_batch(
            outputs["InChI"],
            n_workers=1,
            verbose=False,
        )
        outputs["is_valid"] = ~outputs["normed_InChI"].isna()
        outputs["normed_InChI"] = outputs.normed_InChI.where(
            outputs.is_valid,
            outputs.InChI
        )
        return outputs

    def rescore(
        self,
        InChIs: List[str],
        encoder_outputs: List[Dict[str, torch.Tensor]],
    ):
        pad_sequence = SequenceCollateFunction()
        batch = [
            self.tokenizer(inchi)
            for inchi in InChIs
        ]
        batch = {
            k: pad_sequence([b[k] for b in batch]).to(self.device)
            for k in batch[0].keys()
        }
        outputs = self(
            input_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            encoder_outputs=encoder_outputs,
            use_cache=False,
        )
        scorer = GeneratedInChIScorer()
        scores = scorer(outputs["logits"], batch["next_token_ids"])
        return scores
