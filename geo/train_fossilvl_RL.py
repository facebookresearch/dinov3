import argparse
import json
import os
import re
# Windows can load multiple Intel OpenMP runtimes from different libraries.
# This environment variable allows the process to continue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from geo.dataset import ConversationDataset
from geo.model.hf_fossilvl_wrapper import FossilVLForCausalLM


class RLConversationDataset(Dataset):
    def __init__(self, root: str, annotation: str):
        self.dataset = ConversationDataset(root, annotation)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        conversation = sample["conversation"]
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        return {
            "image": sample["image"],
            "conversation": conversation,
        }


class NWPUCaptioningDataset(Dataset):
    """NWPU captioning dataset with image and 5 reference captions."""
    def __init__(self, root: str, annotation: str):
        self.root = root
        with open(annotation, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(self.root, sample["image_name"].replace("\\", "/"))
        captions = sample.get("captions", [])
        return {
            "image": image_path,
            "captions": captions,  # List of 5 reference captions
        }


def conversation_to_prompt(conversation):
    prompt = []
    for message in conversation:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "user":
            prompt.append(f"User: {content}")
        elif role == "assistant":
            prompt.append(f"Assistant: {content}")
        else:
            prompt.append(f"{role.capitalize()}: {content}")
    prompt.append("Assistant:")
    return "\n".join(prompt)


def compute_reward(response: str, required_patterns: list[str], optional_patterns: list[str], bonus_match: float, penalty_missing: float) -> float:
    text = response
    reward = 0.0

    for pattern in required_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reward += bonus_match
        else:
            reward -= penalty_missing

    for pattern in optional_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reward += bonus_match * 0.25

    return reward


def compute_judge_reward(response: str, prompt: str, judge_model: torch.nn.Module, judge_tokenizer: AutoTokenizer, rubric: str | None = None, reference_captions: list[str] | None = None, max_length: int = 512) -> float:
    system_message = "\
        You are an expert evaluator generating highly discriminative rubrics to assess image caption quality.\
        ## Task \
        Identify the most discriminative criteria that separate excellent captions from weak or flawed ones. Focus,→ on subtle but decisive quality differences that generic rubrics typically miss, while covering all,→ critical dimensions of captioning performance.\
        ## Rules you MUST follow:\
        1. Discriminative Power (Highest Priority)\
        - **Only** include criteria where the student model actually **fails** relative to teacher majority. Do NOT,→ create rubrics for aspects the student model already handles correctly.\
        - Each rubric MUST meaningfully distinguish the weak student caption from teacher-consensus level,→ performance.\
        2. Teacher Consensus as Ground Truth\
        - Ground truth = majority agreement among the five teacher models.\
        - A visual element, relationship, or interpretation is considered correct only if >= 2 teachers describe it,→ accurately.\
        3. Weighting by Severity\
        - 3.0: Critical failures (main subject misidentification, hallucination of major elements, missing,→ essential relationships)\
        - 2.0: Important but non-critical (secondary objects, spatial/contextual accuracy, attribute precision)\
        - 1.0: Minor polish (style fluency, phrasing clarity, minor detail richness)\
        4. Binary & Verifiable\
        - Every criterion must have a clear, objective pass/fail rule that can be verified by directly comparing,→ the student caption against teacher consensus.\
        5. Quality over Quantity\
        - Prefer extremely important and sharp rubrics over many many generic ones.\
        ## Output Requirements: \
        IMPORTANT: Return valid JSON object only, enclosed in triple backticks. Do not include any,→ additional text, explanations, or comments outside the JSON. Escape all quotes within string values,→ using backslash (\"). Do not use single quotes or unescaped double quotes within JSON string values:\
        **JSON Structure:**\
        {\
        \"rubrics\": [\
        {\
        \"criterion\": \"Clear, specific criterion (e.g., Identifies the red bicycle in foreground)\",\
        \"description\": \"Detailed explanation of what this measures and why it matters\",\
        \"evaluation_rule\": \"Concrete rule with clear pass/fail condition\",\
        \"weight\": 1.0|2.0|3.0,\
        \"justification\": \"Explain the gap: what you see in the image, what teachers captured, what weak model,→ missed/got wrong\",\
        \"student_already_met\": \"True or False - Whether the weak model already satisfies this criterion\",\
        \"teacher_consensus\": \"Description of what the majority of reliable teachers agree on for this element\"\
        } \
        ] \
        }"
   
    user_message = f"\
        **Weak Model Output:** \
        {response} \
        **Teacher Model Outputs:** \
        1. **Model 1:** {reference_captions[0] if reference_captions and len(reference_captions) > 0 else 'N/A'}\
        2. **Model 2:** {reference_captions[1] if reference_captions and len(reference_captions) > 1 else 'N/A'}\
        3. **Model 3:** {reference_captions[2] if reference_captions and len(reference_captions) > 2 else 'N/A'}\
        4. **Model 4:** {reference_captions[3] if reference_captions and len(reference_captions) > 3 else 'N/A'}\
        5. **Model 5:** {reference_captions[4] if reference_captions and len(reference_captions) > 4 else 'N/A'}\
        **Task:**\
        1. Carefully examine the image and identify all important visual elements.\
        2. Determine teacher consensus (what the majority of the five teacher models describe correctly).\
        3. Evaluate the weak model's caption across all important dimensions of caption quality: accuracy, completeness, clarity, detail, relationships, and contextual interpretation.\
        4. For each dimension you choose to include, create one targeted binary rubric item with appropriate weight (1.0-3.0) using the required JSON structure.\
        5. Do NOT create rubrics for aspects the weak model already handles correctly.\
        "

    if reference_captions and len(reference_captions) >= 5:

        judge_input = f"System: {system_message}\nUser: {user_message}"
    else:
        judge_input = f"System: {system_message}\nUser: Evaluate the following caption: {response}"
    
    inputs = judge_tokenizer(
        judge_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(judge_model.device)

    outputs = judge_model(**inputs)
    logits = outputs.logits

    if logits.shape[-1] == 1:
        score = torch.sigmoid(logits).squeeze(-1)
    else:
        probs = torch.softmax(logits, dim=-1)
        score = probs[:, -1]

    return float(score.mean().item())


def collate_conversation_fn(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "conversation": [sample["conversation"] for sample in batch],
    }


def collate_nwpu_fn(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "captions": [sample["captions"] for sample in batch],  # List of lists
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed RL training for FossilVL with TRL and Transformers")
    parser.add_argument("--conf", type=str, default="geo/config/base.yaml")
    parser.add_argument("--output_dir", type=str, default="./outputs/fossilvl_rl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--required_pattern", action="append", default=[], help="Regex pattern required in the generated completion.")
    parser.add_argument("--optional_pattern", action="append", default=[], help="Regex pattern that gives bonus if present in completion.")
    parser.add_argument("--penalty_missing", type=float, default=0.5, help="Penalty for each missing required regex match.")
    parser.add_argument("--bonus_match", type=float, default=1.0, help="Reward bonus for each matched required regex pattern.")
    parser.add_argument("--judge_model", type=str, default=None, help="External judge model name or path for rubric-based reward scoring.")
    parser.add_argument("--judge_rubric", type=str, default=None, help="Optional rubric text for judge model scoring.")
    parser.add_argument("--judge_max_length", type=int, default=256, help="Max length for judge model tokenization.")
    parser.add_argument("--dataset_type", type=str, default="conversation", choices=["conversation", "nwpu"], help="Type of dataset to load.")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--use_images", action="store_true", help="Use image inputs with FossilVL generation and forward pass")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = OmegaConf.load(args.conf)

    accelerator = Accelerator()
    device = accelerator.device

    if args.dataset_type == "nwpu":
        train_dataset = NWPUCaptioningDataset(conf.data.root, conf.data.train)
        collate_fn = collate_nwpu_fn
    else:
        train_dataset = RLConversationDataset(conf.data.root, conf.data.train)
        collate_fn = collate_conversation_fn

    sampler = DistributedSampler(train_dataset) if accelerator.num_processes > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
    )

    model = FossilVLForCausalLM.from_fossil_conf(conf)
    tokenizer = model.fossil.decoder.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model = model.to(device)

    judge_model = None
    judge_tokenizer = None
    if args.judge_model:
        judge_model = AutoModelForSequenceClassification.from_pretrained(args.judge_model).to(device)
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    required_patterns = args.required_pattern
    optional_patterns = args.optional_pattern
    if not required_patterns and hasattr(conf, "rl") and getattr(conf.rl, "required_patterns", None) is not None:
        required_patterns = list(conf.rl.required_patterns)
    if not optional_patterns and hasattr(conf, "rl") and getattr(conf.rl, "optional_patterns", None) is not None:
        optional_patterns = list(conf.rl.optional_patterns)

    print(f"Required regex patterns: {required_patterns}")
    print(f"Optional regex patterns: {optional_patterns}")

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_reward = 0.0

        for step, batch in enumerate(train_loader, start=1):
            responses = []
            rewards = []
            reference_captions_list = []
            prompts = []

            if args.dataset_type == "nwpu":
                prompts = ["Provide a concise caption of this image."] * len(batch["image"])
                reference_captions_list = batch["captions"]
            else:
                prompts = [conversation_to_prompt(c) for c in batch["conversation"]]

            for idx, (image_path, prompt) in enumerate(zip(batch["image"], prompts)):
                if args.use_images:
                    response = model.generate([image_path], prompt, num_beams=args.num_beams, do_sample=False, max_new_tokens=args.max_new_tokens)
                else:
                    encoded_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
                    output_ids = model.generate(input_ids=encoded_prompt["input_ids"], attention_mask=encoded_prompt["attention_mask"], max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, do_sample=False)
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                responses.append(response)

                ref_captions = reference_captions_list[idx] if args.dataset_type == "nwpu" else None
                if judge_model is not None and judge_tokenizer is not None:
                    rewards.append(compute_judge_reward(
                        response,
                        prompt,
                        judge_model,
                        judge_tokenizer,
                        rubric=args.judge_rubric,
                        reference_captions=ref_captions,
                        max_length=args.judge_max_length,
                    ))
                else:
                    rewards.append(compute_reward(
                        response,
                        required_patterns=required_patterns,
                        optional_patterns=optional_patterns,
                        bonus_match=args.bonus_match,
                        penalty_missing=args.penalty_missing,
                    ))

            if args.dataset_type == "conversation":
                training_conversations = [
                    conv + [{"role": "assistant", "content": response}]
                    for conv, response in zip(batch["conversation"], responses)
                ]
            else:
                training_conversations = None

            if args.use_images:
                if args.dataset_type == "conversation":
                    model_inputs = model._make_model_inputs(batch["image"], training_conversations)
                    outputs = model(images=batch["image"], conversations=training_conversations)
                else:
                    simple_convs = [
                        [{"role": "user", "content": prompts[i]}, {"role": "assistant", "content": responses[i]}]
                        for i in range(len(responses))
                    ]
                    model_inputs = model._make_model_inputs(batch["image"], simple_convs)
                    outputs = model(images=batch["image"], conversations=simple_convs)

                labels = model_inputs["labels"].to(device)
                logits = outputs.logits
                mask = labels != -100
                labels_for_gather = labels.clone()
                labels_for_gather[~mask] = tokenizer.pad_token_id
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs.gather(-1, labels_for_gather.unsqueeze(-1)).squeeze(-1)
                selected = selected * mask
                logprob_sum = selected.sum(dim=1)
            else:
                if args.dataset_type == "conversation":
                    training_text = [conversation_to_prompt(conv) for conv in training_conversations]
                else:
                    training_text = [f"Image caption: {r}" for r in responses]

                encoded_training = tokenizer(
                    training_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                model_inputs = encoded_training
                outputs = model(
                    input_ids=encoded_training["input_ids"],
                    attention_mask=encoded_training["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                labels = encoded_training["input_ids"].clone()
                mask = labels != tokenizer.pad_token_id
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                selected = selected * mask
                logprob_sum = selected.sum(dim=1)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            policy_loss = -(logprob_sum * reward_tensor).mean()
            value_loss = F.mse_loss(
                outputs["values"].masked_fill(~mask, 0.0),
                reward_tensor.unsqueeze(-1).expand_as(outputs["values"]).masked_fill(~mask, 0.0),
            )
            loss = policy_loss + 0.5 * value_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_reward += sum(rewards)

            if step % args.log_interval == 0:
                avg_loss = epoch_loss / step
                avg_reward = epoch_reward / (step * args.batch_size)
                print(f"Epoch {epoch+1} step {step} loss={avg_loss:.4f} reward={avg_reward:.4f}")

        if accelerator.is_main_process:
            model_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(accelerator.unwrap_model(model).state_dict(), model_path)
            print(f"Saved checkpoint: {model_path}")

    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / "final_model.pt"
        torch.save(accelerator.unwrap_model(model).state_dict(), final_path)
        print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
