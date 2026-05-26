import torch
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .fossilVL import FossilVL


class FossilVLConfig(PretrainedConfig):
    model_type = "fossilvl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FossilVLForCausalLM(PreTrainedModel):
    """A lightweight Hugging Face-style wrapper around the local FossilVL model.

    This wrapper exposes `forward` and `generate` in a way compatible with
    common RL libraries (PPO/TRL). It delegates most work to the existing
    `FossilVL` implementation and adds a small `value_head` used by policy
    optimization algorithms.
    """

    config_class = FossilVLConfig

    def __init__(self, fossil_conf, hf_config: Optional[PretrainedConfig] = None):
        # Create a minimal HF config if not provided
        if hf_config is None:
            hf_config = FossilVLConfig()
        super().__init__(hf_config)

        # underlying multimodal model
        self.fossil = FossilVL(fossil_conf)

        # value head maps decoder hidden states -> scalar values (per token)
        decoder_dim = getattr(self.fossil.decoder, "dim", None)
        if decoder_dim is None:
            # fallback: try to introspect from the HF model if available
            try:
                decoder_dim = self.fossil.decoder.model.config.hidden_size
            except Exception:
                decoder_dim = 1024

        self.value_head = torch.nn.Linear(decoder_dim, 1)

    def _make_model_inputs(self, images: List[Any], conversations: List[Any]):
        device = next(self.fossil.parameters()).device
        image_tensors = self.fossil.encoder.get_image_tensors(images).to(device)
        image_embeddings = self.fossil.encoder(image_tensors, return_grid=self.fossil.use_grid)
        image_embeddings = self.fossil.projection(image_embeddings)

        inputs = self.fossil.decoder.prepare_inputs(conversations).to(device)
        text_embeddings = self.fossil.decoder.get_input_embeds(inputs)
        model_inputs = self.fossil.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
        return model_inputs

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                images: Optional[List[Any]] = None,
                conversations: Optional[List[Any]] = None,
                return_dict: bool = True,
                labels: Optional[torch.Tensor] = None,
                output_hidden_states: bool = True,
                **kwargs,
                ) -> CausalLMOutputWithCrossAttentions:
        """Compute logits and value estimates for the provided multimodal inputs.

        Args:
            input_ids: input token IDs for text-only HF-style forwarding.
            attention_mask: attention mask for text-only HF-style forwarding.
            inputs_embeds: input embeddings for text-only HF-style forwarding.
            images: list of raw images or tensors per batch element.
            conversations: tokenized conversation lists used by the multi-modal FossilVL path.
            labels: optional labels tensor for supervised loss passthrough.

        Returns:
            A `CausalLMOutputWithCrossAttentions`-like object with an extra
            `values` field containing per-token value estimates.
        """

        hf_model = self.fossil.decoder.model

        if images is None or conversations is None:
            # Standard HF-style forward for text-only or tokenizer-based APIs.
            outputs = hf_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            model_inputs = self._make_model_inputs(images, conversations)
            outputs = hf_model.forward(
                inputs_embeds=model_inputs['input_embeddings'].to(device=hf_model.device, dtype=hf_model.dtype),
                attention_mask=model_inputs['attention_mask'].to(device=hf_model.device),
                labels=(model_inputs.get('labels') if labels is None else labels).to(device=hf_model.device) if model_inputs.get('labels') is not None or labels is not None else None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        logits = outputs.logits

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
        else:
            last_hidden = outputs.logits.detach()

        values = self.value_head(last_hidden).squeeze(-1)

        out = CausalLMOutputWithCrossAttentions(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=logits,
            past_key_values=getattr(outputs, 'past_key_values', None),
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
        )

        out['values'] = values
        return out

    def generate(self,
                 images: Optional[List[Any]] = None,
                 prompt: Optional[str] = None,
                 input_ids: Optional[torch.Tensor] = None,
                 num_beams: int = 1,
                 do_sample: bool = False,
                 max_new_tokens: int = 100,
                 **kwargs,
                 ):
        """Expose generate through the wrapped FossilVL implementation."""
        if images is not None and prompt is not None:
            return self.fossil.generate(
                images,
                prompt,
                num_beams=num_beams,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        return self.fossil.decoder.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    @classmethod
    def from_fossil_conf(cls, fossil_conf, hf_config: Optional[PretrainedConfig] = None):
        return cls(fossil_conf, hf_config=hf_config)
