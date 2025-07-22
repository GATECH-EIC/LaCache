import os
import random
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from lacache.kv_cache import LaCache


def generate_prompt_landmark(n_garbage: int, n_garbage_prefix: int, seed: int = 1234):
    """
    Generate a text prompt that hides a pass key in a long document of garbage text.
    
    Args:
        n_garbage (int): Total number of garbage characters to include.
        n_garbage_prefix (int): Number of garbage characters before the hidden key.
        seed (int): Seed for reproducibility.
        
    Returns:
        A tuple of (prompt, passkey) where:
            - prompt: The constructed text including the hidden pass key.
            - passkey: The correct pass key as a string.
    """
    # Use a local random generator for reproducibility without affecting global state
    rng = random.Random(seed)
    
    # Define task and garbage text
    task_description = ("There is important information hidden in a lot of irrelevant text. "
                        "Find it and memorize it. You will be quizzed on this hidden information.")
    garbage_snippet = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    # Multiply garbage text to ensure sufficient length
    garbage_text = garbage_snippet * 5000  
    assert len(garbage_text) >= n_garbage, "Garbage text is shorter than expected."

    # Create prefix and suffix of garbage text
    garbage_prefix = garbage_text[:n_garbage_prefix]
    garbage_suffix = garbage_text[: (n_garbage - n_garbage_prefix)]
    
    # Generate a random pass key
    pass_key = rng.randint(1, 99999)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    
    # Build the complete prompt
    prompt_lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    return "\n".join(prompt_lines), str(pass_key)


def run_single_passkey_test(model, tokenizer, device, n_garbage_prefix: int, n_garbage: int, seed: int):
    """
    Run a single test to retrieve the hidden pass key.
    
    Args:
        model: The language model.
        tokenizer: The corresponding tokenizer.
        device: The device for computation.
        n_garbage_prefix (int): Garbage text length before the key.
        n_garbage (int): Total garbage text length.
        seed (int): Seed for generating a specific prompt.
        
    Returns:
        A tuple (is_correct, token_length) indicating whether the retrieved pass key was correct
        and the total token length of the prompt.
    """
    prompt, answer = generate_prompt_landmark(n_garbage, n_garbage_prefix, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    total_input_tokens = input_ids.shape[-1]
    
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids
    max_length = total_input_tokens + answer_ids.shape[-1]
    
    generation_output = model.generate(
        input_ids=input_ids, 
        do_sample=False,
        max_length=max_length, 
        eos_token_id=tokenizer.eos_token_id,
    )
    # Extract the generated answer tokens from the end of the output
    generated_tokens = generation_output[0, -answer_ids.shape[-1]:].cpu()
    model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    gold_answer = answer
    is_correct = (gold_answer in model_answer)
    
    return is_correct, total_input_tokens


def run_single_passkey_test_lacache(model, tokenizer, device, n_garbage_prefix: int, n_garbage: int, seed: int, span: int, overlap: int, budget: float):
    """
    Run a single test using lacache-llm to retrieve the hidden pass key.
    
    Args:
        model: The language model.
        tokenizer: The corresponding tokenizer.
        device: The device for computation.
        n_garbage_prefix (int): Garbage text length before the key.
        n_garbage (int): Total garbage text length.
        seed (int): Seed for generating a specific prompt.
        
    Returns:
        A tuple (is_correct, token_length) indicating whether the retrieved pass key was correct
        and the total token length of the prompt.
    """
    
    prompt, answer = generate_prompt_landmark(n_garbage, n_garbage_prefix, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    total_input_tokens = input_ids.shape[-1]

    kv_cache = LaCache(cache_size=int(budget*total_input_tokens), start_size=64, span=span, overlap=overlap) # 64 tokens are reserved for questions/system instructions
    
    with torch.no_grad():
        past_key_values = DynamicCache()
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        inputs = model.prepare_inputs_for_generation( 
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True
        )
        answer_ids = tokenizer(answer, return_tensors="pt").input_ids
        max_gen_len = answer_ids.shape[-1]
        outputs = model(**inputs)
        past_key_values = outputs.past_key_values

        legacy_cache = kv_cache(past_key_values.to_legacy_cache())
        past_key_values = DynamicCache.from_legacy_cache(legacy_cache)
        
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]
        
        for _ in range(max_gen_len - 1):
            if cache_position[-1:] + 1 <= legacy_cache[0][0].shape[2]:
                cache_position = cache_position[-1:] + 1
            elif cache_position[-1:] > legacy_cache[0][0].shape[2]:
                cache_position = torch.as_tensor([legacy_cache[0][0].shape[2]], dtype=cache_position.dtype, device=input_ids.device)
            
            inputs = model.prepare_inputs_for_generation(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True
            )
            outputs = model(**inputs)
            past_key_values = outputs.past_key_values

            legacy_cache = kv_cache(past_key_values.to_legacy_cache())
            past_key_values = DynamicCache.from_legacy_cache(legacy_cache)
            
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(pred_token_idx.item())
            if pred_token_idx.item() == tokenizer.eos_token_id:
                break
                
        model_answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
    
    gold_answer = answer
    is_correct = (gold_answer in model_answer)
    
    return is_correct, total_input_tokens


def run_parameter_sweep(args, model, tokenizer):
    """
    Run a grid search over context length and document depth parameters.
    Results are saved as a CSV and a heatmap image.
    
    Args:
        args: Argument object with attributes:
            - max_tokens: Maximum number of tokens to test.
            - interval: Step interval for increasing context length.
            - num_tests: Number of tests per configuration.
        model: The language model.
        tokenizer: The corresponding tokenizer.
    """
    total_points = args.max_tokens // args.interval
    all_results = []

    for point in range(total_points):
        # Calculate total garbage length (adjust multiplier as necessary)
        n_garbage = int(3.75 * (point + 1) * args.interval // 1024 * 1024)
        # Evaluate at 10 different prefix lengths, uniformly distributed over the garbage text
        for n_garbage_prefix in range(0, n_garbage+1, max(n_garbage // 10, 1)):
            passed_tests = 0
            total_tokens = 0
            # Repeat test for multiple seeds
            for seed in tqdm(range(args.num_tests), desc=f"Testing at {n_garbage_prefix} prefix length"):
                is_correct, token_len = run_single_passkey_test(model, tokenizer, model.device,
                                                                n_garbage_prefix, n_garbage, seed)
                passed_tests += int(is_correct)
                total_tokens += token_len
                torch.cuda.empty_cache()

            avg_tokens = total_tokens // args.num_tests
            accuracy = passed_tests / args.num_tests
            depth_ratio = n_garbage_prefix / n_garbage
            print(f"Avg tokens: {avg_tokens}, Depth ratio: {depth_ratio:.2f}, Accuracy: {accuracy:.2f}")
            result = {"Context Length": avg_tokens, "Document Depth": round(depth_ratio * 100, -1), "Score": passed_tests}
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    os.makedirs("./data", exist_ok=True)
    csv_path = f"data/heatmap_{args.model_name.split('/')[-1]}_{args.max_tokens}_{args.interval}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    data = pd.read_csv(csv_path)
    data['Context Length'] = data['Context Length'].apply(lambda x: round(x/1000)*1000)

    context_lengths = data['Context Length'].unique()
    document_depths = data['Document Depth'].unique()
    scores = data.pivot(index='Document Depth', columns='Context Length', values='Score')*100/args.num_tests

    print(f"{args.model_name.split('/')[-1]} average score: ", scores.mean().mean())
    cmap = sns.color_palette("RdYlGn", 10)
    plt.figure(figsize=(24, 8), dpi=600)
    ax = sns.heatmap(scores, annot=True, cmap=cmap, cbar=True, vmin=0, vmax=100, square=True, fmt=".2f",
                cbar_kws={'ticks': np.linspace(0, 100, 6)}, linecolor='white', linewidths=0.6)
    ax.tick_params(axis='both', which='both', direction='out')
    ax.set_yticklabels(document_depths, rotation=0)
    context_labels_k = [f"{int(round(cl/1000))}k" for cl in context_lengths]
    ax.set_xticklabels(context_labels_k, rotation=0)
    ax.xaxis.tick_top()
    plt.ylabel("Depth")
    plt.savefig(f"data/heatmap_{args.model_name.split('/')[-1]}_{args.max_tokens}_{args.interval}.png")
    print(f"Heatmap saved to 'data/heatmap_{args.model_name.split('/')[-1]}_{args.max_tokens}_{args.interval}.csv'")


def run_parameter_sweep_lacache(args, model, tokenizer):
    """
    Run a grid search over context length and document depth parameters using LaCache.
    Results are saved as CSV and heatmap image.
    
    Args:
        args: Argument object with attributes:
            - max_tokens: Maximum number of tokens to test.
            - interval: Step interval for increasing context length.
            - num_tests: Number of tests per configuration.
        model: The language model.
        tokenizer: The corresponding tokenizer.
    """
    total_points = args.max_tokens // args.interval
    all_results = []
    for point in range(total_points):
        # Calculate total garbage length
        n_garbage = int(3.75 * (point + 1) * args.interval // 1024 * 1024)
        # Evaluate at 10 different prefix lengths, uniformly distributed over the garbage text
        for n_garbage_prefix in range(0, n_garbage+1, max(n_garbage // 10, 1)):
            passed_tests = 0
            total_tokens = 0
            # Repeat test for multiple seeds
            for seed in tqdm(range(args.num_tests), desc=f"Testing with LaCache at {n_garbage_prefix} prefix length"):
                is_correct, token_len = run_single_passkey_test_lacache(
                    model, tokenizer, model.device,
                    n_garbage_prefix, n_garbage, seed,
                    span=args.span, overlap=args.overlap, budget=args.budget,
                )
                passed_tests += int(is_correct)
                total_tokens += token_len
                torch.cuda.empty_cache()

            avg_tokens = total_tokens // args.num_tests
            accuracy = passed_tests / args.num_tests
            depth_ratio = n_garbage_prefix / n_garbage
            print(f"LaCache - Avg tokens: {avg_tokens}, Depth ratio: {depth_ratio:.2f}, Accuracy: {accuracy:.2f}")
            result = {"Context Length": avg_tokens, "Document Depth": round(depth_ratio * 100, -1), "Score": passed_tests}
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    os.makedirs("./data_lacache", exist_ok=True)
    csv_path = f"data_lacache/heatmap_{args.model_name.split('/')[-1]}_lacache_{args.max_tokens}_{args.interval}_{args.budget}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    data = pd.read_csv(csv_path)
    data['Context Length'] = data['Context Length'].apply(lambda x: round(x/1000)*1000)

    context_lengths = data['Context Length'].unique()
    document_depths = data['Document Depth'].unique()
    scores = data.pivot(index='Document Depth', columns='Context Length', values='Score')*100/args.num_tests

    print(f"{args.model_name.split('/')[-1]} average score: ", scores.mean().mean())
    cmap = sns.color_palette("RdYlGn", 10)
    plt.figure(figsize=(24, 8), dpi=600)
    ax = sns.heatmap(scores, annot=True, cmap=cmap, cbar=True, vmin=0, vmax=100, square=True, fmt=".2f",
                cbar_kws={'ticks': np.linspace(0, 100, 6)}, linecolor='white', linewidths=0.6)
    ax.tick_params(axis='both', which='both', direction='out')
    ax.set_yticklabels(document_depths, rotation=0)
    context_labels_k = [f"{int(round(cl/1024))}k" for cl in context_lengths]
    ax.set_xticklabels(context_labels_k, rotation=0)
    ax.xaxis.tick_top()
    plt.ylabel("Depth")
    plt.savefig(f"data_lacache/heatmap_{args.model_name.split('/')[-1]}_lacache_{args.max_tokens}_{args.interval}_{args.budget}.png")
    print(f"LaCache heatmap saved to 'data_lacache/heatmap_{args.model_name.split('/')[-1]}_lacache_{args.max_tokens}_{args.interval}_{args.budget}.png'")


def main(args):
    """
    Main function to load the model, tokenizer, and run the passkey retrieval tests.
    
    Args:
        args: An argument object with required attributes:
            - max_tokens, interval, num_tests, model_name.
    """
    set_seed(1234)
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2", 
        use_cache=True,
    )
    if args.enable_lacache:
        from lacache.llama_patch import enable_llama_pos_shift_attention
        enable_llama_pos_shift_attention(model)

    # Enable memory efficient attention if available
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        print("Enabling xformers memory efficient attention")
        model.enable_xformers_memory_efficient_attention()
    
    model.eval()
    
    # Run tests with automatic mixed precision if available
    with torch.cuda.amp.autocast():
        if not args.enable_lacache:
            run_parameter_sweep(args, model, tokenizer)
        else:
            run_parameter_sweep_lacache(args, model, tokenizer)

    # if not args.enable_lacache:
    #     run_parameter_sweep(args, model, tokenizer)
    # else:
    #     run_parameter_sweep_lacache(args, model, tokenizer)


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='model for evaluation')
    parser.add_argument('--max_tokens', type=int, default=4096, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=2948, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--enable_lacache', action='store_true', help='enable lacache for evaluation')
    parser.add_argument("--span", type=int, default=14)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument('--budget', type=float, default=0.5, help='kv cache budget')
    args = parser.parse_args()

    main(args)

