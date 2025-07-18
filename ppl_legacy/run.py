import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from lacache.kv_cache import LaCache


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model.eval()
    return model, tokenizer


def main(args):
    data = load_dataset(args.dataset_name, args.task, split=args.split)
    model, tokenizer = load(args.model_name_or_path)
    device = "cuda"

    if not "llama" in model.config.model_type:
        raise ValueError(f"got {model.config.model_type}")
        
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")

    if args.enable_lacache:
        kv_cache = LaCache(cache_size=args.cache_size, span=args.span, overlap=args.overlap)
        from lacache.llama_patch import enable_llama_pos_shift_attention
        enable_llama_pos_shift_attention(model)
    else:
        kv_cache = None

    if args.dataset_name == "wikitext":
        encodings = tokenizer("".join(data["text"]), return_tensors="pt")
        # torch.save(encodings, 'encodings_wikitext_llama2.pt') # one may first save encodings as pt for a faster loading,
        # torch.save(encodings, 'encodings_wikitext_llama3.pt') # one may first save encodings as pt for a faster loading, 
        # encodings = torch.load('encodings_wikitext_llama2.pt') # and then loading encodings direcntly from pt
        # encodings = torch.load('encodings_wikitext_llama3.pt') # and then loading encodings direcntly from pt 
    elif args.dataset_name == "emozilla/pg19-test":
        encodings = tokenizer("".join(data["text"][:2]), return_tensors="pt") # [:2]: using only first 2 books for a faster tokenization, adjust it for a longer length
    else:
        raise ValueError(f"got {args.dataset_name}")


    num_eval_tokens = 0
    pbar = tqdm(range(0, args.num_eval_tokens + 1))
    nlls = []
    past_key_values = None
    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
        nlls.append(neg_log_likelihood)
        pbar.set_description(f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}")
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens > args.num_eval_tokens:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"ppl: {ppl.item()}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/llama/llama-7b")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--enable_lacache", action="store_true")
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--span", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--num_eval_tokens", type=int, default=None)
    args = parser.parse_args()
    
    main(args)
