# code adapted from https://github.com/Baijiong-Lin/LoRA-Torch/blob/main/examples/Finetune_open_clip_with_LoRA_Torch_on_CIFAR10.ipynb
import loratorch as lora


def apply_lora_attn_mlp(model, conf):
    encoders = []
    if conf.encoder == 'vision':
        encoders.append(model.visual.transformer)

    elif conf.encoder == 'text':    
        encoders.append(model.transformer)
    
    elif conf.encoder == 'both':
        encoders.append(model.visual.transformer)
        encoders.append(model.transformer)
        
    else:
        raise ValueError("Invalid encoder_type. Choose from 'visual', 'text' or 'both'.")

    enable_lora=conf.params # ['q', 'k', 'v', 'o']
    for encoder in encoders:
        for i, resblock in enumerate(encoder.resblocks):
            if hasattr(resblock, 'attn'):
                multihead = resblock.attn
                lora_multihead = lora.MultiheadAttention(r=conf.r,
                                        lora_alpha=conf.alpha,
                                        enable_lora=enable_lora,
                                        embed_dim=multihead.embed_dim,
                                        num_heads=multihead.num_heads,
                                        dropout=multihead.dropout,
                                        bias=True if hasattr(multihead, "in_proj_bias") else False,
                                        add_bias_kv=False if multihead.bias_k==None else True,
                                        add_zero_attn=multihead.add_zero_attn,
                                        kdim=multihead.kdim,
                                        vdim=multihead.vdim,
                                        batch_first=multihead.batch_first)
                lora_multihead.load_state_dict(multihead.state_dict(), strict=False)
                resblock.attn = lora_multihead

    lora.mark_only_lora_as_trainable(model)
    return model