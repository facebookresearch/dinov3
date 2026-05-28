import torch
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig, ProcessorMixin, AutoTokenizer, AutoImageProcessor
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .fossilVL import FossilVL


class FossilVLProcessor(ProcessorMixin):
    # Define quais sub-processadores fazem parte desta classe para salvamento/carregamento automático
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, dim=224, **kwargs):
        # Passa os sub-processadores para o construtor base do Hugging Face
        super().__init__(image_processor, tokenizer, **kwargs)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.dim = dim
        

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        output_data = {}

        processed_images = []
        
        for image in images:
            if type(image[0]) == str:
                im = Image.open(image[0])

            im = self.image_processor(im.convert('RGB'))
            processed_images.append(im)
            print('preprocess', im.shape)

        output_data['pixel_values'] = processed_images
        # print('preprocess', text)
        conversations = []
        for t in text:
            temp = [
                {
                "role": "user",
                "content": t,
                }
            ]
            conversations.append(temp)
        output_data['conversations'] = conversations
        output_data.update(self.tokenizer(text, return_tensors='pt'))
        return output_data
     
    @property
    def default_chat_template(self):
        """Opcional: Define como as mensagens de chat se transformam em strings."""
        # print(self.tokenizer.default_chat_template)
        return self.tokenizer.default_chat_template


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

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                pixel_values: Optional[List[Any]] = None,
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
        if inputs_embeds is None:
            print('wraper forward', pixel_values.shape)
            if type(pixel_values) == List:
                images = torch.stack(pixel_values, dim=0)
            
            image_embeddings = self.fossil.encoder(pixel_values, return_grid=self.fossil.use_grid)
            image_embeddings = self.fossil.projection(image_embeddings)

            inputs = self.fossil.decoder.prepare_inputs(conversations, add_gen_prompt=True)
            # print('INPUTS', inputs)
            text_embeddings = self.fossil.decoder.get_input_embeds(inputs)
            model_inputs = self.fossil.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
                    
            # print('KWARG', kwargs)
            old_mask = kwargs.pop('attention_mask', None)
        
            model_inputs = self.fossil.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
            # print('MODEL INPUTS', model_inputs)

            # Standard HF-style forward for text-only or tokenizer-based APIs.
            outputs = self.fossil.decoder.model.forward(
                inputs_embeds=model_inputs['input_embeddings'].to(old_mask.device),
                attention_mask=model_inputs['attention_mask'].to(old_mask.device),
                labels=labels,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        
        else:
            outputs = self.fossil.decoder.model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
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
                 pixel_values: Optional[List[Any]] = None,
                 prompt: Optional[str] = None,
                 input_ids: Optional[torch.Tensor] = None,
                 num_beams: int = 1,
                 do_sample: bool = False,
                 max_new_tokens: int = 100,
                 conversations: Optional[List[Any]]=None,
                 **kwargs,
                 ):
        """Expose generate through the wrapped FossilVL implementation."""
        images = torch.stack(pixel_values, dim=0)
        image_embeddings = self.fossil.encoder(images, return_grid=self.fossil.use_grid)
        image_embeddings = self.fossil.projection(image_embeddings)

        inputs = self.fossil.decoder.prepare_inputs(conversations, add_gen_prompt=True)
        # print('INPUTS', inputs)
        text_embeddings = self.fossil.decoder.get_input_embeds(inputs)
        model_inputs = self.fossil.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
                
        # print('KWARG', kwargs)
        old_mask = kwargs.pop('attention_mask', None)
        return self.fossil.decoder.model.generate(
            inputs_embeds=model_inputs['input_embeddings'].to(old_mask.device),
            attention_mask=model_inputs['attention_mask'].to(old_mask.device),
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    @classmethod
    def from_fossil_conf(cls, fossil_conf, hf_config: Optional[PretrainedConfig] = None):
        return cls(fossil_conf, hf_config=hf_config)
