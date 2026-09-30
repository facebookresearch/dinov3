"""
GRPO fine-tuning of FossilVL with an LLM judge as reward model.

Reward of a generated description, from the same yes/no claims and judge code as eval/judge_reliability.py:
    recall = mean P(yes) over the sample's annotated labels
    false  = mean P(yes) over labels the sample does not have: labels of the same category from other samples
             and its main/secondary constituents in the other role (resampled every step, shared by the
             completions of a prompt)
    reward = recall - false_weight * false
Listing every possible constituent raises recall but is paid back through the false claims.

Process layout (torchrun, one process per GPU): on every node the first --policy_gpus_per_node local ranks train
the captioning model (gradients averaged among them), the other local ranks each hold a replica of the judge.
Every step:
    1. policy ranks sample --num_generations descriptions for each of their --batch_size images
    2. every (description, claim) question is gathered on all ranks and each judge rank scores its share
    3. the scores are gathered back; policy ranks compute group-normalised advantages and take one step

The captioning model starts from an SFT output folder (config.yaml + <ckpt>_checkpoint.pt, as written by
trainFossilVL.py). --out_dir receives config.yaml and last_checkpoint.pt in the same format, so generate.py
works on it unchanged, plus log.jsonl, samples.jsonl and rl_state.pt (for --resume).

Labels are matched to the training conversations by their reference description (the assistant turn), using a
file with 'refs' and 'labels' lists such as eval/todos.json.

    torchrun --nproc_per_node 4 geo/train_fossilvl_grpo.py --model <sft dir> --labels geo/eval/todos.json \
        --out_dir <rl dir>
See geo/slurm/train_grpo.sbatch.
"""
import argparse
import copy
import datetime
import json
import os
import random
import sys
import time
from collections import defaultdict

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

GEO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GEO)
sys.path.insert(0, os.path.join(GEO, 'eval'))
from dataset import ConversationDataset  # noqa: E402
from eval_labels_llm import HFJudge, build_claim, score_pairs  # noqa: E402
from judge_reliability import label_pools, normalize_labels, sample_claims  # noqa: E402
from model.fossilVL import FossilVL  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------------------------------- data

class RLDataset(torch.utils.data.Dataset):
    """Image, prompt, reference and annotated labels of every conversation of a split that has labels."""

    def __init__(self, conf, split_path, labels_path, campo=None):
        base = ConversationDataset(conf.data.root, split_path, groups=conf.data.get('groups'), campo=campo)
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        by_ref = defaultdict(set)
        for ref, labels in zip(data['refs'], data['labels']):
            by_ref[ref.strip()].add(json.dumps(normalize_labels(labels), ensure_ascii=False, sort_keys=True))
        self.pools = label_pools(normalize_labels(labels) for labels in data['labels'])

        self.samples = []
        skipped = defaultdict(int)
        for image, conversation in zip(base.image, base.conversation):
            conversation = json.loads(conversation)
            reference = conversation[1]['content'].strip()
            options = by_ref.get(reference, set())
            if len(options) != 1:
                skipped['no labels for the reference' if not options else 'reference with several label sets'] += 1
                continue
            labels = json.loads(next(iter(options)))
            if not labels:
                skipped['empty labels'] += 1
                continue
            self.samples.append({'image': image, 'prompt': conversation[0]['content'], 'reference': reference,
                                 'labels': labels})
        self.skipped = dict(skipped)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# -------------------------------------------------------------------------------------------------- policy

def load_fossil(conf, ckpt_path):
    """FossilVL with the SFT weights, built as in generate.py."""
    model = FossilVL(conf)
    if not hasattr(model.decoder.model, 'peft_config') and conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model


def encoder_size(conf, image_path):
    """Input size used in SFT: the larger size for images at least that large, else the smaller one."""
    sizes = conf.encoder.size
    if not OmegaConf.is_list(sizes):
        return None
    if len(sizes) == 1:
        return sizes[0]
    with Image.open(image_path) as image:
        return max(sizes) if min(image.size) >= max(sizes) else min(sizes)


@torch.no_grad()
def image_features(model, conf, image_path):
    """Frozen encoder output for one image, [1, encoder dim]."""
    size = encoder_size(conf, image_path)
    tensors = (model.encoder.get_image_tensors([image_path]) if size is None
               else model.encoder.get_image_tensors([image_path], size=size))
    return model.encoder(tensors, return_grid=model.use_grid)


def prompt_embeddings(decoder, projection, features, prompt):
    """Chat prompt with the projected image tokens merged in, [1, prompt length, decoder dim]."""
    ids = decoder.prepare_inputs([[{'role': 'user', 'content': prompt}]], add_gen_prompt=True)
    ids = ids.to(features.device)
    merged = decoder.merge_inputs(projection(features), decoder.get_input_embeds(ids), ids)
    return merged['input_embeddings']


def eos_ids(decoder):
    ids = decoder.model.generation_config.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids)


@torch.no_grad()
def sample_completions(decoder, embeds, args):
    """num_generations sampled completions of one prompt: token ids [G, T] and mask [G, T] (up to the first eos,
    included; completions that hit max_new_tokens keep every token)."""
    ids = decoder.model.generate(
        inputs_embeds=embeds, attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
        max_new_tokens=args.max_new_tokens, num_return_sequences=args.num_generations,
        eos_token_id=eos_ids(decoder), pad_token_id=decoder.tokenizer.pad_token_id)
    is_eos = torch.isin(ids, torch.tensor(eos_ids(decoder), device=ids.device))
    first_eos = torch.where(is_eos.any(1), is_eos.int().argmax(1), torch.full_like(ids[:, 0], ids.shape[1]))
    mask = (torch.arange(ids.shape[1], device=ids.device)[None] <= first_eos[:, None]).long()
    return ids, mask, ~is_eos.any(1)


def token_logps(decoder, projection, features, prompt, ids, mask, temperature):
    """Log-probability of every completion token under the model, [G, T] (padding positions included)."""
    embeds = prompt_embeddings(decoder, projection, features, prompt)
    n, length = ids.shape
    inputs = torch.cat([embeds.expand(n, -1, -1), decoder.get_input_embeds(ids)], dim=1)
    attention = torch.cat([torch.ones(n, embeds.shape[1], dtype=mask.dtype, device=mask.device), mask], dim=1)
    # the last length + 1 positions predict the completion tokens (the final one predicts past the end)
    logits = decoder.model(inputs_embeds=inputs, attention_mask=attention, use_cache=False,
                           logits_to_keep=length + 1).logits[:, :-1].float()
    return torch.gather((logits / temperature).log_softmax(-1), 2, ids[..., None]).squeeze(-1)


# ---------------------------------------------------------------------------------------------- distributed

def setup_distributed(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank, local_world = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    if args.policy_gpus_per_node >= local_world:
        sys.exit(f'{local_world} processes per node leave no GPU for the judge; '
                 f'lower --policy_gpus_per_node ({args.policy_gpus_per_node})')
    device = torch.device(f'cuda:{local_rank % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    backend = args.backend or ('nccl' if device.type == 'cuda' else 'gloo')
    # judges spend minutes loading and scoring while the policy ranks wait in a collective
    dist.init_process_group(backend, timeout=datetime.timedelta(minutes=args.timeout_minutes))

    policy_ranks = [r for r in range(world) if r % local_world < args.policy_gpus_per_node]
    judge_ranks = [r for r in range(world) if r not in policy_ranks]
    return {'rank': rank, 'world': world, 'device': device, 'is_policy': rank in policy_ranks,
            'policy_ranks': policy_ranks, 'judge_ranks': judge_ranks,
            'policy_group': dist.new_group(policy_ranks)}


def exchange(env, questions, scores_fn):
    """All ranks: gathers the questions of the policy ranks, lets each judge rank score its share, returns the
    scores of this rank's own questions (policy ranks) or None (judge ranks)."""
    gathered = [None] * env['world']
    dist.all_gather_object(gathered, questions)
    flat = [q for part in gathered if part for q in part]

    mine = {}
    if not env['is_policy']:
        index = env['judge_ranks'].index(env['rank'])
        share = list(range(index, len(flat), len(env['judge_ranks'])))
        mine = dict(zip(share, scores_fn([flat[i] for i in share])))
    results = [None] * env['world']
    dist.all_gather_object(results, mine)

    if not env['is_policy']:
        return None
    scores = {}
    for part in results:
        scores.update(part)
    offset = sum(len(part or []) for part in gathered[:env['rank']])
    return [scores[offset + i] for i in range(len(questions))]


def make_scorer(judge, batch_size):
    return lambda questions: score_pairs(judge, questions, batch_size)


# ------------------------------------------------------------------------------------------------ training

def rewards_of(claims, probs, args):
    """recall, false-claim rate and reward of one completion from the P(yes) of its claims."""
    if args.reward_threshold is not None:
        probs = [float(p >= args.reward_threshold) for p in probs]
    pos = [p for (_, _, gold, _), p in zip(claims, probs) if gold == 1]
    neg = [p for (_, _, gold, _), p in zip(claims, probs) if gold == 0]
    recall = sum(pos) / len(pos) if pos else 0.0
    false = sum(neg) / len(neg) if neg else 0.0
    return recall, false, recall - args.false_weight * false


def batch_indices(n_samples, step, env, args, steps_per_epoch):
    """This policy rank's samples for a step: a seeded shuffle per epoch, split among the policy ranks."""
    order = list(range(n_samples))
    random.Random(args.seed + step // steps_per_epoch).shuffle(order)
    per_step = args.batch_size * len(env['policy_ranks'])
    start = (step % steps_per_epoch) * per_step + env['policy_ranks'].index(env['rank']) * args.batch_size
    return order[start:start + args.batch_size]


def average_gradients(params, env, bucket_size=2 ** 26):
    """Averages gradients over the policy ranks, in flat buckets of about bucket_size values."""
    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    n_policy = len(env['policy_ranks'])
    if n_policy == 1:
        return
    via_cpu = dist.get_backend(env['policy_group']) == 'gloo'  # gloo cannot reduce CUDA tensors on every build
    start = 0
    while start < len(grads):
        end, size = start, 0
        while end < len(grads) and (end == start or size + grads[end].numel() <= bucket_size):
            size += grads[end].numel()
            end += 1
        flat = torch.cat([g.flatten() for g in grads[start:end]])
        reduced = flat.cpu() if via_cpu else flat
        dist.all_reduce(reduced, group=env['policy_group'])
        flat = reduced.to(flat.device) / n_policy
        offset = 0
        for g in grads[start:end]:
            g.copy_(flat[offset:offset + g.numel()].view_as(g))
            offset += g.numel()
        start = end


def policy_step(step, batch, model, ref, conf, pools, optimizer, params, env, args):
    device = env['device']
    decoder, projection = model.decoder, model.projection
    rng = random.Random(args.seed * 1_000_003 + step * 1_000 + env['rank'])
    t0 = time.time()

    groups, questions = [], []
    for sample in batch:
        features = image_features(model, conf, sample['image']).to(device)
        with torch.no_grad():
            embeds = prompt_embeddings(decoder, projection, features, sample['prompt'])
        ids, mask, truncated = sample_completions(decoder, embeds, args)
        texts = [decoder.tokenizer.decode(ids[i][mask[i].bool()], skip_special_tokens=True).strip()
                 for i in range(len(ids))]
        claims = sample_claims(sample['labels'], pools, rng, args.neg_ratio)
        claim_texts = [build_claim(category, label) for category, label, _, _ in claims]
        groups.append({'sample': sample, 'features': features, 'ids': ids, 'mask': mask, 'texts': texts,
                       'claims': claims, 'truncated': truncated, 'first_question': len(questions)})
        questions += [(text, claim) for text in texts for claim in claim_texts]
    t_generate = time.time() - t0

    t0 = time.time()
    scores = exchange(env, questions, None)
    t_judge = time.time() - t0

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    total_tokens = max(sum(int(g['mask'].sum()) for g in groups), 1)
    stats = defaultdict(list)
    for g in groups:
        n_claims = len(g['claims'])
        first = g['first_question']
        per_completion = [rewards_of(g['claims'], scores[first + i * n_claims:first + (i + 1) * n_claims], args)
                          for i in range(len(g['texts']))]
        rewards = torch.tensor([r[2] for r in per_completion], device=device)
        advantages = rewards - rewards.mean()
        if args.scale_rewards:
            advantages = advantages / (rewards.std() + 1e-4)
        g['rewards'] = rewards.tolist()

        logps = token_logps(decoder, projection, g['features'], g['sample']['prompt'], g['ids'], g['mask'],
                            args.temperature)
        # single on-policy update: the ratio is 1 in value, its gradient is the policy gradient
        loss = -advantages[:, None] * torch.exp(logps - logps.detach())
        if args.beta > 0:
            with torch.no_grad():
                ref_logps = token_logps(ref['decoder'], ref['projection'], g['features'], g['sample']['prompt'],
                                        g['ids'], g['mask'], args.temperature)
            delta = ref_logps - logps
            kl = torch.exp(delta) - delta - 1  # k3 estimator of KL(policy || reference), per token
            loss = loss + args.beta * kl
            stats['kl'].append(float((kl.detach() * g['mask']).sum() / g['mask'].sum()))
        ((loss * g['mask']).sum() / total_tokens).backward()

        stats['reward'] += [r[2] for r in per_completion]
        stats['recall'] += [r[0] for r in per_completion]
        stats['false_claims'] += [r[1] for r in per_completion]
        stats['reward_std_in_group'].append(float(rewards.std()))
        stats['completion_tokens'] += g['mask'].sum(1).tolist()
        stats['truncated'] += g['truncated'].float().tolist()

    average_gradients(params, env)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
    optimizer.step()
    t_update = time.time() - t0

    local = dict(stats)
    local.update(grad_norm=[float(grad_norm)], t_generate=[t_generate], t_judge=[t_judge], t_update=[t_update])
    return local, groups


def save(model, optimizer, step, args):
    tmp = os.path.join(args.out_dir, 'last_checkpoint.pt.tmp')
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, os.path.join(args.out_dir, 'last_checkpoint.pt'))
    torch.save({'step': step, 'optimizer': optimizer.state_dict()}, os.path.join(args.out_dir, 'rl_state.pt'))


def run_policy(env, args):
    conf = OmegaConf.load(os.path.join(args.model, 'config.yaml'))
    split = {'train': conf.data.train, 'val': conf.data.val, 'test': conf.data.test}[args.split]
    dataset = RLDataset(conf, split, args.labels, campo=args.campo)
    steps_per_epoch = len(dataset) // (args.batch_size * len(env['policy_ranks']))
    if steps_per_epoch == 0:
        sys.exit(f'{len(dataset)} samples with labels: fewer than one step')
    total_steps = args.max_steps or args.epochs * steps_per_epoch

    model = load_fossil(conf, os.path.join(args.model, f'{args.ckpt}_checkpoint.pt'))
    ref = {'decoder': copy.deepcopy(model.decoder), 'projection': copy.deepcopy(model.projection)}
    for module in ref.values():
        module.to(env['device']).eval().requires_grad_(False)
    model.to(env['device']).eval()  # eval: no dropout, so sampling and log-probs see the same model
    model.encoder.requires_grad_(False)
    model.projection.requires_grad_(not args.freeze_projection)
    if not hasattr(model.decoder.model, 'peft_config'):
        model.decoder.requires_grad_(True)  # with LoRA only the adapters stay trainable
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start = 0
    state_path = os.path.join(args.out_dir, 'rl_state.pt')
    if args.resume and os.path.exists(state_path):
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'last_checkpoint.pt'), map_location='cpu'))
        state = torch.load(state_path, map_location='cpu')
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']

    is_main = env['rank'] == env['policy_ranks'][0]
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        out_conf = copy.deepcopy(conf)
        out_conf.save_path = args.out_dir
        OmegaConf.save(out_conf, os.path.join(args.out_dir, 'config.yaml'))
        with open(os.path.join(args.out_dir, 'rl_args.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2)
        n_train = sum(p.numel() for p in params)
        print(f'{len(dataset)} samples with labels (skipped: {dataset.skipped}); {steps_per_epoch} steps per epoch, '
              f'steps {start}-{total_steps}; {n_train:,} trainable parameters; policy ranks {env["policy_ranks"]}, '
              f'judge ranks {env["judge_ranks"]}', flush=True)
    dist.broadcast_object_list([start, total_steps], src=env['policy_ranks'][0])  # tells the judges how long
    torch.manual_seed(args.seed + env['rank'])

    for step in range(start, total_steps):
        batch = [dataset[i] for i in batch_indices(len(dataset), step, env, args, steps_per_epoch)]
        local, groups = policy_step(step, batch, model, ref, conf, dataset.pools, optimizer, params, env, args)

        gathered = [None] * len(env['policy_ranks'])
        dist.all_gather_object(gathered, local, group=env['policy_group'])
        if is_main:
            merged = defaultdict(list)
            for part in gathered:
                for k, v in part.items():
                    merged[k] += v
            record = {'step': step + 1, **{k: sum(v) / len(v) for k, v in merged.items()}}
            record['t_generate'] = max(merged['t_generate'])
            record['t_judge'] = max(merged['t_judge'])
            with open(os.path.join(args.out_dir, 'log.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in record.items()),
                  flush=True)
            if args.log_samples_every and (step + 1) % args.log_samples_every == 0:
                g = groups[0]
                with open(os.path.join(args.out_dir, 'samples.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'step': step + 1, 'image': g['sample']['image'],
                                        'reference': g['sample']['reference'], 'labels': g['sample']['labels'],
                                        'completions': g['texts'], 'rewards': g['rewards']},
                                       ensure_ascii=False) + '\n')
            if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
                save(model, optimizer, step + 1, args)


def run_judge(env, args):
    # device_map loads the weights straight onto this GPU instead of through CPU RAM (~65 GB per 32B judge)
    judge = HFJudge(args.judge, load_in_4bit=args.judge_load_in_4bit, device=str(env['device']),
                    device_map=str(env['device']))
    if set(judge.yes_ids) & set(judge.no_ids):
        sys.exit(f'yes/no token ids overlap for {args.judge}')
    score = make_scorer(judge, args.judge_batch_size)
    steps = [None, None]
    dist.broadcast_object_list(steps, src=env['policy_ranks'][0])
    start, total_steps = steps
    for _ in range(start, total_steps):
        exchange(env, None, score)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', required=True, help='SFT output folder with config.yaml and <ckpt>_checkpoint.pt')
    p.add_argument('--ckpt', default='best', choices=['best', 'last'])
    p.add_argument('--labels', default='geo/eval/todos.json', help="json with 'refs' and 'labels' lists")
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--campo', default=None)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--resume', action='store_true', help='continue from out_dir/rl_state.pt if present')

    p.add_argument('--judge', default='Qwen/Qwen2.5-32B-Instruct')
    p.add_argument('--judge_load_in_4bit', action='store_true')
    p.add_argument('--judge_batch_size', default=32, type=int)
    p.add_argument('--policy_gpus_per_node', default=1, type=int, help='the rest of each node runs the judge')

    p.add_argument('--batch_size', default=4, type=int, help='images per policy rank per step')
    p.add_argument('--num_generations', default=8, type=int, help='completions per image (GRPO group)')
    p.add_argument('--epochs', default=1, type=int)
    p.add_argument('--max_steps', default=None, type=int, help='overrides --epochs')
    p.add_argument('--lr', default=1e-6, type=float)
    p.add_argument('--weight_decay', default=0.0, type=float)
    p.add_argument('--max_grad_norm', default=1.0, type=float)
    p.add_argument('--beta', default=0.04, type=float, help='KL penalty to the SFT model (0 disables it)')
    p.add_argument('--freeze_projection', action='store_true')
    p.add_argument('--no_scale_rewards', dest='scale_rewards', action='store_false',
                   help='advantages = reward - group mean, without dividing by the group std')
    p.add_argument('--temperature', default=1.0, type=float)
    p.add_argument('--top_p', default=1.0, type=float)
    p.add_argument('--max_new_tokens', default=160, type=int)

    p.add_argument('--neg_ratio', default=1.0, type=float, help='absent-label claims per annotated label')
    p.add_argument('--false_weight', default=1.0, type=float, help='weight of the false-claim rate in the reward')
    p.add_argument('--reward_threshold', default=None, type=float,
                   help='count claims as 0/1 at this P(yes) instead of using P(yes) directly')

    p.add_argument('--save_every', default=50, type=int)
    p.add_argument('--log_samples_every', default=10, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--backend', default=None, help='nccl on GPUs by default')
    p.add_argument('--timeout_minutes', default=60, type=int)
    return p.parse_args()


def main(args):
    if args.num_generations < 2:
        sys.exit('GRPO compares completions of the same image: --num_generations must be at least 2')
    env = setup_distributed(args)
    try:
        if env['is_policy']:
            run_policy(env, args)
        else:
            run_judge(env, args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(parse_args())
