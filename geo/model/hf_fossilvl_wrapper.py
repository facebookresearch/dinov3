import os
import torch
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig, ProcessorMixin, ImageProcessingMixin
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .fossilVL import FossilVL


class ImageProcessorWrapper(ImageProcessingMixin):
    def __init__(self, preprocess_method, size=224):
        super().__init__()
        self.preprocess_method = preprocess_method
        self.size = size

    def __call__(self, images, **kwargs):
        images = self.preprocess_method(images, self.size, **kwargs)
        return images


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
        chat_template = """
            {% for message in messages %}
            {% if message.role == "user" %}
            <|im_start|>user
            {% for item in message.content %}
            {% if item.type == "image" %}<|vision_start|><|vision_end|>
            {% elif item.type == "text" %}{{ item.text }}{% endif %}
            {% endfor %}<|im_end|>

            {% elif message.role == "assistant" %}
            <|im_start|>assistant
            <think>

            </think>

            {{ message.content }}<|im_end|>

            {% endif %}
            {% endfor %}
            """
        self.chat_template = chat_template

    def __call__(self, images=None, text=None, return_tensors="pt", add_generation_prompt=True, **kwargs):
        output_data = {}
        processed_images = []
        print('processor call', text)
        if add_generation_prompt:
            text = [t+"<|im_start|>assistant\n<think>\n\n</think>\n" for t in text]

        for image in images:
            im = Image.open(image[0]).convert('RGB')
            im = self.image_processor(im)
            processed_images.append(im)

        output_data['pixel_values'] = torch.stack(processed_images)
        output_data.update(self.tokenizer(text, return_tensors='pt', )) #add_generation_prompt=True
        return output_data
     
    # @property
    # def default_chat_template(self):
    #     """Opcional: Define como as mensagens de chat se transformam em strings."""
    #     return self.tokenizer.default_chat_template


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
        ckpt = torch.load(os.path.join(fossil_conf.save_path, f'last_checkpoint.pt'), map_location=torch.device('cpu'))
        self.fossil.load_state_dict(ckpt)
        
        # value head maps decoder hidden states -> scalar values (per token)
        decoder_dim = getattr(self.fossil.decoder, "dim", None)
        
        if decoder_dim is None:
            # fallback: try to introspect from the HF model if available
            try:
                decoder_dim = self.fossil.decoder.model.config.hidden_size
            except Exception:
                decoder_dim = 1024

        self.value_head = torch.nn.Linear(decoder_dim, 1)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        pixel_values=None,
        **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
        }

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                pixel_values: Optional[List[Any]] = None,
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
            labels: optional labels tensor for supervised loss passthrough.

        Returns:
            A `CausalLMOutputWithCrossAttentions`-like object with an extra
            `values` field containing per-token value estimates.
        """
        input_text = self.fossil.decoder.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print('input text forward:', input_text) 

        image_embeddings = None
        if pixel_values is not None:
            image_embeddings = self.fossil.encoder(pixel_values, return_grid=self.fossil.use_grid)
            image_embeddings = self.fossil.projection(image_embeddings)
        
        text_embeddings = None
        if input_ids is not None:
            text_embeddings = self.fossil.decoder.get_input_embeds(input_ids)

        if image_embeddings is not None and text_embeddings is not None and input_ids is not None:
            model_inputs = self.fossil.decoder.merge_inputs(image_embeddings, text_embeddings, input_ids)
            print('forward if:', image_embeddings.shape, text_embeddings.shape, input_ids.shape, model_inputs['input_embeddings'].shape)
            
            outputs = self.fossil.decoder.model.forward(
                inputs_embeds=model_inputs['input_embeddings'],
                attention_mask=model_inputs['attention_mask'],
                input_ids=None,
                labels=labels,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        
        else:
            print('forward else', inputs_embeds.shape, pixel_values.shape, input_ids.shape)
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
                 input_ids: Optional[torch.Tensor] = None,
                 num_beams: int = 1,
                 do_sample: bool = False,
                 max_new_tokens: int = 100,
                 **kwargs,
                 ):
        """Expose generate through the wrapped FossilVL implementation."""
        if type(pixel_values) == List:
            pixel_values = torch.stack(pixel_values, dim=0)
        
        input_text = self.fossil.decoder.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print('input text generate:', input_text)    
        image_embeddings = None
        if pixel_values is not None:
            image_embeddings = self.fossil.encoder(pixel_values, return_grid=self.fossil.use_grid)
            image_embeddings = self.fossil.projection(image_embeddings)
        
        text_embeddings = None
        if input_ids is not None:
            text_embeddings = self.fossil.decoder.get_input_embeds(input_ids)

        if image_embeddings is not None and text_embeddings is not None and input_ids is not None:
            model_inputs = self.fossil.decoder.merge_inputs(image_embeddings, text_embeddings, input_ids)        
            old_mask = kwargs.pop('attention_mask', None)
            print('generate if:', image_embeddings.shape, text_embeddings.shape, input_ids.shape, model_inputs['input_embeddings'].shape)
            output = self.fossil.decoder.model.generate(
                inputs_embeds=model_inputs['input_embeddings'].to(old_mask.device),
                attention_mask=model_inputs['attention_mask'].to(old_mask.device),
                input_ids=None,
                num_beams=num_beams,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            print("decoded:\n", self.fossil.decoder.tokenizer.batch_decode(output))
            return output
        
        else:
            print('generate else', pixel_values.shape, input_ids.shape)
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
