from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


if __name__ == '__main__':
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    # model_name = 'Qwen/Qwen2-1.5B-Instruct'
    # model_name = "Qwen/Qwen2-0.5B"
    # model_name = "facebook/opt-350m"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_dir = '/nethome/recpinfo/users/fibz/data/checkpoints/qwen3-finetune'
    dataset = '/nethome/recpinfo/users/fibz/data/dataset/publico-COMPLETO.txt'

    tokenizer = AutoTokenizer.from_pretrained(model_name, )

    test_data = load_dataset('text',
                             data_files=dataset,
                             encoding='utf8',
                             cache_dir=output_dir,
                             split='train[:10%]')
    
    train_data = load_dataset('text',
                              data_files=dataset,
                              encoding='utf8',
                              cache_dir=output_dir,
                              split='train[10%:]')

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    trainArgs = SFTConfig(
        fp16=True,
        logging_steps=10000,
        logging_strategy='steps',
        learning_rate=0.001,
        output_dir=output_dir,
        save_strategy='steps',
        save_steps=10000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        overwrite_output_dir=True,
        resume_from_checkpoint=True,
        save_total_limit=10,
        ddp_find_unused_parameters=False,
        # fsdp="full_shard",
    )

    trainer = SFTTrainer(
        model_name,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=config,
        args=trainArgs,
    )
    
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_dir)