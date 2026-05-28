import argparse
import json
from torch.utils.data import  Dataset
import os
import re
import base64
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from trl import GRPOTrainer, GRPOConfig
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.hf_fossilvl_wrapper import FossilVLForCausalLM, FossilVLConfig, FossilVLProcessor
from transformers import AutoConfig, AutoModel, ImageProcessingMixin

class ImageProcessorWrapper(ImageProcessingMixin):
    def __init__(self, preprocess_method, size=224):
        super().__init__()
        self.preprocess_method = preprocess_method
        self.size = size

    def __call__(self, images, **kwargs):
        # Redireciona a chamada para o seu método original
        images = self.preprocess_method(images, self.size, **kwargs)
        print('processorWraper', images.shape)
        return images


class NWPUCaptioningDataset(Dataset):
    """NWPU captioning dataset with image and 5 reference captions."""
    def __init__(self, root: str, annotation: str, prompt):
        self.root = root
        self.prompt = prompt
        with open(annotation, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(self.root, sample["image_name"].replace("\\", "/"))
        captions = sample['captions']
        return {
            "image": image_path,
            "captions": captions,  # List of 5 reference captions
            "prompt": self.prompt,
        }

def rubric_reward(image, captions, completions, completion_ids, prompts, **kwargs)->list[float]:
    '''
        the reward function must return a list of floats, the rewards computed for each completion
    '''
    system_message = "\
        You are an expert evaluator generating highly discriminative rubrics to assess image caption quality.\
        ## Task \
        Identify the most discriminative criteria that separate excellent captions from weak or flawed ones. Focus on subtle but decisive quality differences that generic rubrics typically miss, while covering all critical dimensions of captioning performance.\
        ## Rules you MUST follow:\
        1. Discriminative Power (Highest Priority)\
        - **Only** include criteria where the student model actually **fails** relative to teacher majority. Do NOT create rubrics for aspects the student model already handles correctly.\
        - Each rubric MUST meaningfully distinguish the weak student caption from teacher-consensus level performance.\
        2. Teacher Consensus as Ground Truth\
        - Ground truth = majority agreement among the five teacher models.\
        - A visual element, relationship, or interpretation is considered correct only if >= 2 teachers describe it accurately.\
        3. Weighting by Severity\
        - 3.0: Critical failures (main subject misidentification, hallucination of major elements, missing essential relationships)\
        - 2.0: Important but non-critical (secondary objects, spatial/contextual accuracy, attribute precision)\
        - 1.0: Minor polish (style fluency, phrasing clarity, minor detail richness)\
        4. Binary & Verifiable\
        - Every criterion must have a clear, objective pass/fail rule that can be verified by directly comparing the student caption against teacher consensus.\
        5. Quality over Quantity\
        - Prefer extremely important and sharp rubrics over many many generic ones.\
        ## Output Requirements: \
        IMPORTANT: Return valid JSON object only, enclosed in triple backticks. Do not include any additional text, explanations, or comments outside the JSON. Escape all quotes within string values,→ using backslash (\"). Do not use single quotes or unescaped double quotes within JSON string values:\
        **JSON Structure:**\
        {\
        \"rubrics\": [\
        {\
        \"criterion\": \"Clear, specific criterion (e.g., Identifies the red bicycle in foreground)\",\
        \"description\": \"Detailed explanation of what this measures and why it matters\",\
        \"evaluation_rule\": \"Concrete rule with clear pass/fail condition\",\
        \"weight\": 1.0|2.0|3.0,\
        \"justification\": \"Explain the gap: what you see in the image, what teachers captured, what weak model missed/got wrong\",\
        \"student_already_met\": \"True or False - Whether the weak model already satisfies this criterion\",\
        \"teacher_consensus\": \"Description of what the majority of reliable teachers agree on for this element\"\
        } \
        ] \
        }"
    
    # print("COMPLETIONS", completions)
    for i in range(len(completions)):
        # print(i, completions[i])
        user_message = f"\
            **Weak Model Output:** \
            {completions[i]} \
            **Teacher Model Outputs:** \
            1. **Model 1:** {captions[i][0]}\
            2. **Model 2:** {captions[i][1]}\
            3. **Model 3:** {captions[i][2]}\
            4. **Model 4:** {captions[i][3]}\
            5. **Model 5:** {captions[i][4]}\
            **Task:**\
            1. Carefully examine the image and identify all important visual elements.\
            2. Determine teacher consensus (what the majority of the five teacher models describe correctly).\
            3. Evaluate the weak model's caption across all important dimensions of caption quality: accuracy, completeness, clarity, detail, relationships, and contextual interpretation.\
            4. For each dimension you choose to include, create one targeted binary rubric item with appropriate weight (1.0-3.0) using the required JSON structure.\
            5. Do NOT create rubrics for aspects the weak model already handles correctly.\
            "
        with open(image[i], 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
       
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": user_message
                 },
                 {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"    
                     }
                 }
             ]}
        ]
    
    return [float(0.0)]*len(completions)

def collate_nwpu_fn(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "captions": [sample["captions"] for sample in batch],  # List of lists
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed RL training for FossilVL with TRL and Transformers")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dataset_type", type=str, default="conversation", choices=["conversation", "nwpu"], help="Type of dataset to load.")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = OmegaConf.load(os.path.join(args.model, 'config.yaml'))
  
    train_dataset = NWPUCaptioningDataset(os.path.join(os.path.dirname(args.dataset), 'images'), args.dataset, 'Provide a concise caption of this satellite image')
    model = FossilVLForCausalLM.from_fossil_conf(conf)
    print(model)
    
    training_args = GRPOConfig(
        per_device_train_batch_size=args.batch_size,
        num_generations=args.batch_size, #number of completions
        num_train_epochs=args.epochs,
        output_dir=os.path.join(args.model, 'RL_finetune'), 
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=rubric_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=FossilVLProcessor(
            ImageProcessorWrapper(model.fossil.encoder.preprocess, conf.encoder.size), 
            model.fossil.decoder.tokenizer, 
            model.fossil.decoder.prepare_inputs
        ),
  
    )
    trainer.train()
    
if __name__ == "__main__":
    main()
