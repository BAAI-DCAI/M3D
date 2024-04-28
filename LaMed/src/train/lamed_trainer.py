import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class LaMedTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))