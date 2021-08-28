import torch

import nncomp.registry as R


@R.ModelRegistry.add
class ImageCaptioningModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: dict,
        encoder: dict,
        decoder: dict,
    ):
        super().__init__()
        self.tokenizer = R.PreprocessorRegistry.get_from_params(**tokenizer)
        self.encoder = R.ModuleRegistry.get_from_params(**encoder)
        self.decoder = R.ModuleRegistry.get_from_params(
            vocab_size=len(self.tokenizer),
            encoder_num_features=self.encoder.num_features,
            **decoder,
        )

    def __call__(self, **batch):
        encoder_outputs = self.encode(batch["image"])
        outputs = self.decoder(
            input_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            **encoder_outputs,
        )
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
