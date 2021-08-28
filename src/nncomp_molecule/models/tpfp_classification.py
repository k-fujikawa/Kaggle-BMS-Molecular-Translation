from typing import List

import torch
from loguru import logger

import nncomp.registry as R


@R.ModelRegistry.add
class TPFPClassificationModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: dict,
        encoder: dict,
        decoder: dict,
        out_tasks: List[str] = ["is_GT", "levenshtein"],
        pretrained_model: str = None,
    ):
        super().__init__()
        self.tokenizer = R.PreprocessorRegistry.get_from_params(**tokenizer)
        self.encoder = R.ModuleRegistry.get_from_params(**encoder)
        self.decoder = R.ModuleRegistry.get_from_params(
            vocab_size=len(self.tokenizer),
            encoder_num_features=self.encoder.num_features,
            **decoder,
        )
        if pretrained_model is not None:
            ckpt = torch.load(pretrained_model, map_location="cpu")
            logger.info(self.load_state_dict(ckpt))
        self.out_tasks = out_tasks
        hidden_size = self.decoder.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, len(out_tasks)),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def valid_inchi_token_id(self):
        token_id = self.tokenizer.token_to_id("<EOS>")  # 2
        return torch.tensor(token_id, device=self.device)

    @property
    def invalid_inchi_token_id(self):
        token_id = self.tokenizer.token_to_id("<UNK>")  # 3
        return torch.tensor(token_id, device=self.device)

    def __call__(self, **batch):
        # invalidの場合にEOSをUNKに変更
        eos_token_idx = batch["attention_mask"].sum(dim=1) - 1
        batch_size = len(eos_token_idx)
        eos_token_ids = torch.full_like(
            eos_token_idx,
            self.valid_inchi_token_id,
        )
        eos_token_ids = eos_token_ids.where(
            batch["is_valid"],
            self.invalid_inchi_token_id
        )
        batch["token_ids"][range(batch_size), eos_token_idx] = eos_token_ids

        # encode & decode
        encoder_outputs = self.encode(batch["image"])
        outputs = self.decoder(
            input_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            **encoder_outputs,
        )

        # head
        h = outputs.hidden_states[-1][range(batch_size), eos_token_idx]
        y = self.head(h)

        for i, task in enumerate(self.out_tasks):
            outputs[f"y_{task}"] = y[:, i]

        return outputs

    def encode(self, image):
        h = self.encoder.forward_features(image)
        if isinstance(h, dict):
            return h
        if len(h.shape) == 4:
            h = h.permute(0, 2, 3, 1)
            h = h.reshape(h.shape[0], -1, h.shape[3])
        return dict(
            encoder_hidden_states=h,
            encoder_attention_mask=None,
        )
