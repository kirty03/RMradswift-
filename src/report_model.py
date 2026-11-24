"""report_model.py - the RadSwift model: one chest X-ray in, a draft report and an abnormal score out.

  image -> ViT (image encoder) -> 197 image tokens of size 768
        -> project (one linear layer) -> 197 tokens of size 512
        -> T5 (text model) -> report text
  first image token -> abnormal_head (one linear layer) -> abnormal score

ViT and T5 are public pretrained models. `project` and `abnormal_head` are new.
Fine-tuning (Step 4) updates all four parts on the Indiana chest X-ray reports.
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, ViTConfig, ViTModel

VIT_NAME = "google/vit-base-patch16-224-in21k"   # general-purpose image model (Dosovitskiy et al., 2021)
T5_NAME = "t5-small"                             # text-to-text model (Raffel et al., 2020)
MAX_TOKENS = 160                                 # longest report we train on or generate, in tokens


class XrayReportModel(nn.Module):
    def __init__(self, vit_config=None, t5_config=None):
        """No arguments: load the public pretrained ViT and T5 (needs internet once).
        With configs: build empty networks, to be filled by load_state_dict (offline demo)."""
        super().__init__()
        if vit_config is None:
            self.vit = ViTModel.from_pretrained(VIT_NAME, add_pooling_layer=False)
            self.t5 = T5ForConditionalGeneration.from_pretrained(T5_NAME)
        else:
            self.vit = ViTModel(vit_config, add_pooling_layer=False)
            self.t5 = T5ForConditionalGeneration(t5_config)
        vit_dim = self.vit.config.hidden_size        # 768
        t5_dim = self.t5.config.d_model              # 512
        self.project = nn.Linear(vit_dim, t5_dim)    # image features -> the size T5 expects
        self.abnormal_head = nn.Linear(vit_dim, 1)   # normal vs abnormal score

    def encode_image(self, pixel_values):
        patches = self.vit(pixel_values=pixel_values).last_hidden_state    # (batch, 197, 768)
        abnormal_logit = self.abnormal_head(patches[:, 0]).squeeze(-1)     # first token summarises the image
        return self.project(patches), abnormal_logit                       # (batch, 197, 512), (batch,)

    def forward(self, pixel_values, labels=None):
        """labels = token ids of the true report, with padding set to -100.
        T5 builds its own decoder inputs by shifting labels one step right.
        NEVER pass labels as decoder_input_ids: that was the 2025 bug (the model saw the answer)."""
        image_tokens, abnormal_logit = self.encode_image(pixel_values)
        out = self.t5(inputs_embeds=image_tokens, labels=labels)
        return out.loss, abnormal_logit

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens=MAX_TOKENS, num_beams=4):
        """Write a report for each image. Returns token ids and abnormal probabilities."""
        image_tokens, abnormal_logit = self.encode_image(pixel_values)
        mask = torch.ones(image_tokens.shape[:2], dtype=torch.long, device=image_tokens.device)
        encoded = self.t5.encoder(inputs_embeds=image_tokens, attention_mask=mask)
        token_ids = self.t5.generate(encoder_outputs=encoded, attention_mask=mask,
                                     max_new_tokens=max_new_tokens, num_beams=num_beams,
                                     no_repeat_ngram_size=3, early_stopping=True)
        return token_ids, torch.sigmoid(abnormal_logit)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")      # the demo laptop runs on CPU; that is fast enough for one image


def save_model(model, tokenizer, folder):
    """Save weights, the two network descriptions and the tokenizer, so the demo needs no internet."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "radswift_model.pt")
    model.vit.config.save_pretrained(folder / "vit_config")
    model.t5.config.save_pretrained(folder / "t5_config")
    tokenizer.save_pretrained(folder / "tokenizer")


def load_model(folder, device="cpu"):
    """Load the fine-tuned model saved by save_model()."""
    folder = Path(folder)
    model = XrayReportModel(ViTConfig.from_pretrained(folder / "vit_config"),
                            T5Config.from_pretrained(folder / "t5_config"))
    model.load_state_dict(torch.load(folder / "radswift_model.pt", map_location="cpu"))
    tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")
    return model.to(device).eval(), tokenizer
