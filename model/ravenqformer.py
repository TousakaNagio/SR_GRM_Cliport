from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip2_models.blip2_t5_instruct import Blip2T5Instruct

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F


class RavenQformer(Blip2Qformer):
    def __init__(self, config):
        super().__init__(
            img_size=config.custom.img_size,
            **dict(config.model)
        )
        
        # TODO: reconstuct head
        
        self.img_h = int(config.custom.image_height)
        self.img_w = int(config.custom.image_width)

        self.sim_loss = nn.CosineEmbeddingLoss()
        self.embed_dim = config.custom.embed_dim
        self.sim_head = nn.Linear(self.Qformer.config.hidden_size, self.embed_dim)
        self.affordance_head = nn.Linear(self.Qformer.config.hidden_size, self.img_h * self.img_w)
    
    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["init_image"]
        text = samples["text_input"]
        
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        Qformer_atts = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
        query_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(
            self.affordance_head(query_output.last_hidden_state[:, : query_tokens.size(1), :]), dim=-1
        )
        logit = image_feats # [4, 32, 51200]
        return logit.reshape((logit.size(0), logit.size(1), -1, self.img_h))
        
        