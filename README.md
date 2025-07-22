<div align="center">
<h1>LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models</h1>
</div>

<p align="center">
    <a href="https://arxiv.org/pdf/2507.14204">
        <img alt="Paper" src="https://img.shields.io/badge/paper-link-blue?logo=quicklook" />
    </a>
    <a href="https://arxiv.org/abs/2507.14204">
        <img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2507.14204-B31B1B?logo=arxiv" />
    </a><br>
    On <a href="#on-perplexity"> Perplexity</a> (Wikitext2/PG19) | On <a href="#on-longbench">LongBench</a> | On <a href="#on-needle-in-a-haystack">Needle-In-A-Haystack</a>
</p>

## ⚙️ Installation

> [!NOTE]
> [transformers](https://github.com/huggingface/transformers) had made a major change on kv cache implementation since version 4.36.0. Please use [ppl_legacy](./ppl_legacy) if you are using transformers < 4.36.0

```bash
# if you are using transformers >= 4.36.0
conda create -n lacache python=3.8.20
conda activate lacache
pip install torch==2.4.1 transformers==4.43.4
conda env update --file environment.yml

# if you are using transformers < 4.36.0
conda create -n lacache_legacy python=3.8.20
conda activate lacache_legacy
pip install torch==2.4.1 transformers==4.33.0
conda env update --file environment_legacy.yml
```

## On Perplexity

* `cd ppl/` for the normal version or `cd ppl_legacy/` for the legacy version
  
### 📑 Evaluation

```bash
# Examples on how to use
export HF_TOKEN="YOUR_HUGGINGFACE_KEY"

# Evaluated on wikitext2 dataset
python -u run.py --model_name_or_path meta-llama/Meta-Llama-3-8B --enable_lacache \
    --cache_size 256 --span 16 --overlap 0 --num_eval_tokens 1024

# Evaluated on a different model
python -u run.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --enable_lacache \
    --cache_size 256 --span 16 --overlap 0 --num_eval_tokens 1024

# Evaluated with a different generation length
python -u run.py --model_name_or_path meta-llama/Meta-Llama-3-8B --enable_lacache \
    --cache_size 256 --span 16 --overlap 0 --num_eval_tokens 8192

# Evaluated with a different cache size
python -u run.py --model_name_or_path meta-llama/Meta-Llama-3-8B --enable_lacache \
    --cache_size 512 --span 16 --overlap 0 --num_eval_tokens 1024

# Evaluated with different ladder hyperparameters
python -u run.py --model_name_or_path meta-llama/Meta-Llama-3-8B --enable_lacache \
    --cache_size 256 --span 8 --overlap 4 --num_eval_tokens 1024

# Evaluated on pg19 dataset
python -u run.py --model_name_or_path meta-llama/Meta-Llama-3-8B --dataset_name emozilla/pg19-test --task default \
    --split test --enable_lacache --cache_size 256 --span 16 --overlap 0 --num_eval_tokens 1024

# Evaluated without LaCache
python -u run.py --model_name_or_path meta-llama/Meta-Llama-3-8B --num_eval_tokens 1024
```

* To evaluate on newer models such as Llama-3.1 and Llama-3.2, please use [ppl](./ppl) instead of [ppl_legacy](./ppl_legacy)
* To evaluate with flash-attention, please specify `attn_implementation="flash_attention_2"` (see the load function in [ppl/run.py](./ppl/run.py))


## On LongBench

* `cd LongBench/` for the normal version or `cd LongBench_legacy/` for the legacy version

### 📑 Evaluation

```bash
# Examples on how to use
export HF_TOKEN="YOUR_HUGGINGFACE_KEY"

# Evaluated on meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=0 python pred.py --model meta-llama/Llama-2-7b-chat-hf --method lacache --budget 0.5 --span 16 --overlap 0
python eval.py --model meta-llama/Llama-2-7b-chat-hf --method lacache --budget 0.5

# Evaluated with a different cache budget and ladder hyperparameters
CUDA_VISIBLE_DEVICES=0 python pred.py --model meta-llama/Llama-2-7b-chat-hf --method lacache --budget 0.25 --span 9 --overlap 3
python eval.py --model meta-llama/Llama-2-7b-chat-hf --method lacache --budget 0.25

# Evaluated on a different model
CUDA_VISIBLE_DEVICES=0 python pred.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --method lacache --budget 0.5 --span 9 --overlap 7
python eval.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --method lacache --budget 0.5

CUDA_VISIBLE_DEVICES=0 python pred.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --method lacache --budget 0.25 --span 9 --overlap 7
python eval.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --method lacache --budget 0.25

# Evaluated without LaCache
CUDA_VISIBLE_DEVICES=0 python pred.py --model meta-llama/Llama-2-7b-chat-hf
python eval.py --model meta-llama/Llama-2-7b-chat-hf
```

## On Needle-In-A-Haystack

* `cd niah/` 
  
### 📑 Evaluation

```bash
# Examples on how to use
export HF_TOKEN="YOUR_HUGGINGFACE_KEY"

# Evaluated with a context lenght of 128*1024=131072
python pred.py --model_name meta-llama/Llama-3.2-3B-Instruct --max_tokens 131072 --interval 16384 --num_tests 50 --enable_lacache \
    --span 14 --overlap 0 --budget 0.5

# Evaluated with different maximum context length and evaluation interval 
python pred.py --model_name meta-llama/Llama-3.2-3B-Instruct --max_tokens 65536 --interval 32768 --num_tests 50 --enable_lacache \
    --span 14 --overlap 0 --budget 0.5

# Evaluated with different number of tests
python pred.py --model_name meta-llama/Llama-3.2-3B-Instruct --max_tokens 131072 --interval 16384 --num_tests 100 --enable_lacache \
    --span 14 --overlap 0 --budget 0.5

# Evaluated without LaCache
python pred.py --model_name meta-llama/Llama-3.2-3B-Instruct --max_tokens 131072 --interval 16384 --num_tests 50
```

* Only support `transformers >= 4.36.0` currently, *i.e.*, no legacy version
* Decrease `--max_tokens` if gpu memory is insufficent



## 💬 Acknowledgments
This code is built upon <a href="https://github.com/huggingface/transformers">transformers</a>, <a href="https://github.com/THUDM/LongBench">LongBench</a>, <a href="https://github.com/mit-han-lab/streaming-llm">streaming-llm</a>, and <a href="https://github.com/jzhang38/LongMamba">LongMamba</a>. We thank the contributors of these open-source projects.

## ✨ Citation
```bibtex
@inproceedings{shilacache,
  title={LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models},
  author={Shi, Dachuan and Fu, Yonggan and Yuan, Xiangchi and Yu, Zhongzhi and You, Haoran and Li, Sixu and Dong, Xin and Kautz, Jan and Molchanov, Pavlo and Lin, Yingyan Celine},
  booktitle={Forty-second International Conference on Machine Learning},
  year = {2025}
}
```
