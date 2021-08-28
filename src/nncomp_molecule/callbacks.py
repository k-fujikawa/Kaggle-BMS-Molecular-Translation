import os
from pathlib import Path

import catalyst.dl
import pandas as pd
import torch
import Levenshtein
from catalyst.dl import CallbackOrder, Callback
from tqdm import tqdm
from loguru import logger

import nncomp
import nncomp.registry as R
import nncomp_molecule
from nncomp_molecule.encoders.swin import checkpoint_filter_fn


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


@R.CallbackRegistry.add
class GenerationCallback(catalyst.dl.Callback):
    def __init__(
        self,
        loader: str,
    ):
        super().__init__(order=CallbackOrder.External)
        self.loader = loader

    @torch.no_grad()
    def on_stage_end(self, runner: catalyst.dl.IRunner):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        logger.info("Start to inference")
        model = runner.model.eval()
        model.load_state_dict(
            torch.load(runner.logdir / "best.pth", map_location=runner.device),
        )
        generation_config = nncomp_molecule.generators.GenerationConfig(
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
        generator = nncomp_molecule.generators.EnsenmbleBeamSearchGenerator(
            tokenizer=model.tokenizer,
            config=generation_config,
            models=[model],
            weights=[1],
        )
        generator.to(runner.device)
        outputs_df = pd.DataFrame()
        loader = runner.loaders[self.loader]
        for batch in tqdm(loader, desc=f"{self.loader} (generation)"):
            batch = nncomp.utils.to_device(batch, runner.device)
            batch["InChI_GT"] = batch["InChI"]
            batch_size = len(batch["image_id"])
            batch_outputs_df = pd.DataFrame({
                k: batch[k]
                for k in ["image_id", "InChI", "InChI_GT"]
                if k in batch
            })

            with torch.no_grad(), torch.cuda.amp.autocast():
                input_ids = torch.full(
                    (batch_size, 1),
                    model.tokenizer.token_to_id("<BOS>"),
                    device=runner.device,
                )
                encoder_outputs = generator.encode(batch["image"])
                outputs = generator.generate(
                    input_ids=input_ids,
                    encoder_outputs=encoder_outputs,
                )
                outputs = generator.postprocess(outputs)
                normed_score = generator.rescore(
                    InChIs=outputs.normed_InChI,
                    encoder_outputs=encoder_outputs,
                )

            batch_outputs_df["score"] = outputs.log_likelihood
            batch_outputs_df["InChI"] = outputs.InChI
            batch_outputs_df["is_valid"] = outputs.is_valid
            batch_outputs_df["normed_InChI"] = outputs.normed_InChI
            batch_outputs_df["normed_score"] = normed_score

            batch_outputs_df["levenshtein"] = [
                Levenshtein.distance(InChI, InChI_GT)
                for InChI, InChI_GT
                in batch_outputs_df[["InChI", "InChI_GT"]].values
            ]
            outputs_df = outputs_df.append(batch_outputs_df, ignore_index=True)

        outputs_df.to_csv(runner.logdir.parent / "valid_beam=1.csv", index=False)
        print(outputs_df.levenshtein.mean())


@R.CallbackRegistry.add
class SwinTransformerResumeTrainingCallback(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_filename: str = "best.pth",
        strict: bool = False,
        disable: bool = False,
    ):
        super().__init__(order=CallbackOrder.external)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_filename = checkpoint_filename
        self.strict = strict
        self.disable = disable

    def on_stage_start(self, runner: catalyst.dl.IRunner):
        if self.disable:
            return
        path = self.checkpoint_dir / runner.stage / self.checkpoint_filename
        checkpoint = catalyst.utils.load_checkpoint(path)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        checkpoint = {
            k: v
            for k, v in checkpoint.items()
            if "attn_mask" not in k
        }
        logger.info(runner.model.load_state_dict(
            checkpoint,
            strict=self.strict,
        ))
