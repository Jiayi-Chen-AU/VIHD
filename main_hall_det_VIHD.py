import os

# Please use your huggingface cache dir
HF_CACHE = "xxx/huggingface"    

# Must be set before importing transformers / datasets to take effect
os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HOME"] = HF_CACHE

import csv
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta, get_semantic_ids
from SeEntLib.demo import get_sentence_semantic_entropy_w_semantic_ids
from decoder_attn.QwenAttnAdapter import QwenAttnAdapter
from decoder_attn.Qwen3AttnAdapter import Qwen3AttnAdapter

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


MODEL_PATHS = {
    'lingshu': f"{HF_CACHE}/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
    'hulu':    f"{HF_CACHE}/models--ZJU-AI4H--Hulu-Med-4B/snapshots/caec06b4421e6ede4a1927155c7c25ffb92f1a5c",
}
SLAKE_IMG_DIR = f"{HF_CACHE}/SLAKE/imgs"


# ═════════════════════════════════════════════════════════════════════
# Utils
# ═════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"[Seed] Random seed set to {seed}")


def get_messages(prompt, image):
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a medical image analysis expert."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]},
    ]


def get_sequence_log_logits(token_ids, scores, special_token_ids):
    log_logits = torch.log(F.softmax(scores, dim=-1))
    selected = torch.gather(log_logits, 2, token_ids.unsqueeze(-1)).squeeze(-1)
    mask = ~torch.isin(token_ids, special_token_ids)
    return [logits[m].cpu() for logits, m in zip(selected, mask)]


# ═════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════
def load_model(model_name, device):
    model_id = MODEL_PATHS[model_name]
    if model_name == 'lingshu':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map=device, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=True)
        decoder_layers, adapter_cls = model.language_model.layers, QwenAttnAdapter
    elif model_name == 'hulu':
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map=device, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        decoder_layers, adapter_cls = model.model.layers, Qwen3AttnAdapter
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Replace self-attention with the adapter
    for i, layer in enumerate(decoder_layers):
        adap = adapter_cls(layer.self_attn.config, i)
        adap.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = adap.to(dtype=model.dtype, device=device)

    return model, processor, decoder_layers


def build_inputs(image, prompt, processor, model, model_name):
    messages = get_messages(prompt, image)
    if model_name == 'lingshu':
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        return dict(inputs)
    # hulu
    inputs = processor(conversation=messages, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs


def get_vision_range(input_ids, model_name):
    if model_name == 'lingshu':
        start = torch.where(input_ids == 151652)[0] + 1   # <|vision_start|>
        end   = torch.where(input_ids == 151653)[0]       # <|vision_end|>
    else:  # hulu
        img_idx = torch.where(input_ids == 151669)[0]
        start, end = img_idx[0], img_idx[-1] + 1
    return start, end


def set_vision_range(decoder_layers, start, end):
    for layer in decoder_layers:
        layer.prefix_len = None
        if layer.self_attn.vision_start is None:
            layer.self_attn.vision_start = start
            layer.self_attn.vision_end = end


def set_prune(decoder_layers, layer_mask=None):
    """Enable pruning on layers where layer_mask is True; layer_mask=None disables pruning on all layers."""
    for i, layer in enumerate(decoder_layers):
        layer.self_attn.prune = bool(layer_mask[i]) if layer_mask is not None else False


def decode(token_ids, processor, model_name):
    if model_name == 'hulu':
        return processor.batch_decode(token_ids, skip_special_tokens=True, use_think=False)
    return [s.strip() for s in processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True)]


def generate(inputs, model, processor, temperature=1.0, output_attentions=False, min_new_tokens=1):
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=200, min_new_tokens=min_new_tokens, do_sample=True, temperature=temperature,
            top_p=0.9, num_beams=1, use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True,
            output_attentions=output_attentions, output_logits=False,
        )


def sample_answers(image_input, prompt, base_inputs, input_len, num_samples, aug_transform,
                   processor, model, model_name):
    """Sample num_samples answers. If aug_transform is None, no augmentation is applied and base_inputs is reused."""
    def make_inputs():
        if aug_transform is None:
            return base_inputs
        return build_inputs(aug_transform(image_input), prompt, processor, model, model_name)

    if model_name == 'hulu':
        # hulu: generate one sample at a time (batch=1)
        token_ids_list, scores_list, answers = [], [], []
        for _ in range(num_samples):
            out = generate(make_inputs(), model, processor)
            token_ids_list.append(out.sequences.squeeze(0))
            scores_list.append(torch.stack(out.scores))                 # (T, 1, V)
            answers.append(decode(out.sequences, processor, model_name)[0])
        token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        scores = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf).squeeze(2)  # (N, T, V)
    else:
        # lingshu: concatenate into one batch and generate at once
        inputs_list = [make_inputs() for _ in range(num_samples)]
        batch = {k: torch.cat([x[k] for x in inputs_list], dim=0) for k in inputs_list[0]}
        out = generate(batch, model, processor)
        token_ids = out.sequences[:, input_len:]
        scores = torch.stack(out.scores, dim=1)                         # (N, T, V)
        answers = decode(token_ids, processor, model_name)
    return token_ids, scores, answers


# ═════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════
def load_vqa_dataset(dataset_name, mode, q_type):
    is_close = lambda x: x["answer"].lower() in ["yes", "no"]
    if dataset_name in ['vqa-rad', 'vqa-med-2019']:
        hf_name = {"vqa-rad": "flaviagiammarino/vqa-rad",
                   "vqa-med-2019": "claudioreeves/imageclef-vqa-med-2019"}[dataset_name]
        ds = load_dataset(hf_name, split=mode)
        if q_type == "open":
            ds = ds.filter(lambda x: not is_close(x))
        elif q_type == "close":
            ds = ds.filter(is_close)
    elif dataset_name == 'slake':
        ds = load_dataset("BoKelvin/SLAKE", split=mode).filter(lambda x: x["q_lang"].lower() == "en")
        if q_type == 'open':
            ds = ds.filter(lambda x: x["answer_type"] == "OPEN")
        elif q_type == 'close':
            ds = ds.filter(lambda x: x["answer_type"] != "OPEN")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds


def get_sample(dataset, idx, dataset_name):
    item = dataset[idx]
    if dataset_name == 'slake':
        image = Image.open(os.path.join(SLAKE_IMG_DIR, item['img_name']))
        type_idx = item["answer_type"].lower()
    else:
        image = item['image']
        type_idx = "close" if item["answer"].lower() in ["yes", "no"] else "open"
    return image.convert('RGB'), item['question'], item['answer'], type_idx


# ═════════════════════════════════════════════════════════════════════
# VIHD score
# ═════════════════════════════════════════════════════════════════════
def compute_vihd(question, gen_answer, sam_answers, sam_answers_prune,
                 token_ids, scores, token_ids_prune, scores_prune,
                 special_ids, entailment_model, num_samples, alpha=1.0):
    texts = [f'{question} {gen_answer}'] + [f'{question} {r}' for r in sam_answers + sam_answers_prune]
    semantic_ids = get_semantic_ids(texts, model=entailment_model, strict_entailment=False)
    ids_normal = semantic_ids[1:num_samples + 1]
    ids_prune  = semantic_ids[num_samples + 1:2 * num_samples + 1]

    log_logits       = get_sequence_log_logits(token_ids, scores, special_ids)
    log_logits_prune = get_sequence_log_logits(token_ids_prune, scores_prune, special_ids)
    _, se_dist       = get_sentence_semantic_entropy_w_semantic_ids(log_logits, ids_normal)
    _, se_dist_prune = get_sentence_semantic_entropy_w_semantic_ids(log_logits_prune, ids_prune)

    align       = torch.zeros(max(semantic_ids) + 1)
    align_prune = torch.zeros_like(align)
    align[torch.unique(torch.tensor(ids_normal))]      = se_dist
    align_prune[torch.unique(torch.tensor(ids_prune))] = se_dist_prune

    cosine = F.cosine_similarity(align.unsqueeze(0), align_prune.unsqueeze(0)).squeeze()
    if cosine < 0.95:
        merged = torch.softmax(align + alpha * (align - align_prune), dim=-1)
    else:
        merged = (align + alpha * align_prune) / (1.0 + alpha)

    return -torch.sum(merged * torch.log(merged + 1e-10)).item()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
def main_vihd(model_name, dataset_name, q_type, mode, augment=False, num_samples=10):
    device = torch.device("cuda:0")
    model, processor, decoder_layers = load_model(model_name, device)
    num_layers = len(decoder_layers)
    dataset = load_vqa_dataset(dataset_name, mode, q_type)
    entailment_model = EntailmentDeberta(device=device)
    special_ids = torch.tensor(processor.tokenizer.all_special_ids, device=device)

    img_trans_ori = transforms.Resize((512, 512))
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]) if augment else None

    suffix = '_aug' if augment else ''
    csv_file = f'det_outputs_{mode}/{dataset_name}_{model_name}_vihd_{q_type}{suffix}.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['img_idx', 'question', 'ref_answer', 'gen_answer', 'VIHD', 'answer_type'])

    total_time = 0.0
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='inference'):
            image, question, ref_answer, type_idx = get_sample(dataset, idx, dataset_name)
            image_input = img_trans_ori(image)
            prompt = 'Answer this question as concisely as possible based on the provide images: ' + question

            start_time = time.time()

            inputs = build_inputs(image_input, prompt, processor, model, model_name)
            input_len = inputs["input_ids"].shape[-1]
            vision_start, vision_end = get_vision_range(inputs['input_ids'][0], model_name)
            set_vision_range(decoder_layers, vision_start, vision_end)

            # ── Step 1: Low-temperature generation + collect attention ───
            outputs = generate(inputs, model, processor, temperature=0.1, output_attentions=True, min_new_tokens=0)
            gen_ids = outputs.sequences if model_name == 'hulu' else outputs.sequences[:, input_len:]
            gen_answer = decode(gen_ids, processor, model_name)[0]

            # ── Step 2: VDP — identify visual-dependent layers ──────────
            visual_attn = torch.zeros(num_layers, device=device)
            for token_attn in outputs.attentions[1:-1]:
                for j, layer_attn in enumerate(token_attn):
                    visual_attn[j] += layer_attn[0, :, 0, vision_start:vision_end].sum()

            window_size = num_layers // 2
            start_idx = torch.argmax(visual_attn.unfold(0, window_size, 1).sum(dim=1)).item()
            layer_mask = torch.zeros(num_layers, dtype=torch.bool)
            layer_mask[start_idx:start_idx + window_size] = True

            sample_kwargs = dict(image_input=image_input, prompt=prompt, base_inputs=inputs, input_len=input_len,
                                 num_samples=num_samples, aug_transform=aug_transform,
                                 processor=processor, model=model, model_name=model_name)

            # ── Step 3: Normal sampled responses ────────────────────────
            token_ids, scores, sam_answers = sample_answers(**sample_kwargs)

            # ── Step 4: VID — sample with visual layers pruned ──────────
            set_prune(decoder_layers, layer_mask)
            token_ids_prune, scores_prune, sam_answers_prune = sample_answers(**sample_kwargs)
            set_prune(decoder_layers, None)

            # ── Step 5: VIHD ─────────────────────────────────────────
            VIHD = compute_vihd(
                question, gen_answer, sam_answers, sam_answers_prune,
                token_ids.to(device), scores.to(device), token_ids_prune.to(device), scores_prune.to(device),
                special_ids, entailment_model, num_samples)

            total_time += time.time() - start_time

            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([idx, question, ref_answer, gen_answer, VIHD, type_idx])

            del outputs, token_ids, scores, token_ids_prune, scores_prune, visual_attn
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    print(f"Average inference time per sample: {total_time / len(dataset):.2f} s")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vqa-rad", choices=["vqa-rad", "slake", "vqa-med-2019"])
    parser.add_argument("--model", type=str, default="lingshu", choices=["lingshu", "hulu"])
    parser.add_argument("--type", type=str, default="open")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--augment", action="store_true", help="Apply augmentation to the input images for sampling (off by default)")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=815)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    main_vihd(model_name=args.model, dataset_name=args.dataset, q_type=args.type, mode=args.mode,
              augment=args.augment, num_samples=args.num_samples)