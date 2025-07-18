<div align="center">
<h1>LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models</h1>
</div>

<p align="center">
    <a href="https://github.com/GATECH-EIC/LaCache/blob/main/assets/LaCache.pdf">
        <img alt="Paper" src="https://img.shields.io/badge/paper-link-blue?logo=quicklook" />
    </a>
    <a href="">
        <img alt="ArXiv" src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-B31B1B?logo=arxiv" />
    </a><br>
</p>

## On PPL

### ⚙️ Installation

> [!NOTE]
> [transformers](https://github.com/huggingface/transformers) had made a major change on kv cache implementation since version 4.36.0. Please use [ppl_legacy](./ppl_legacy) if you are using transformers < 4.36.0.

```bash
# transformers < 4.36.0
cd ppl_legacy
conda create -n ppl_legacy python=3.8.20
conda activate ppl_legacy
pip install torch==2.4.1 transformers==4.33.0
conda env update --file environment.yml

# transformers >= 4.36.0
cd ppl
conda create -n ppl python=3.8.20
conda activate ppl
pip install torch==2.4.1 transformers==4.36.2
conda env update --file environment.yml
```

### 📑 Evaluation

```bash
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

* To evaluate newer models such as Llama-3.1 and Llama-3.2, please use [ppl](./ppl) instead of [ppl_legacy](./ppl_legacy)
* To evaluate with flash-attention, please specify `attn_implementation="flash_attention_2" (see the load function in [ppl/run.py](./ppl/run.py))

## 💬 Acknowledgments
This code is built upon <a href="https://github.com/huggingface/transformers">transformers</a> and <a href="https://github.com/mit-han-lab/streaming-llm">streaming-llm</a>. We thank the contributors of these open-source projects.

## ✨ Citation
```bibtex
@inproceedings{shilacache,
  title={LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models},
  author={Shi, Dachuan and Fu, Yonggan and Yuan, Xiangchi and Yu, Zhongzhi and You, Haoran and Li, Sixu and Dong, Xin and Kautz, Jan and Molchanov, Pavlo and Lin, Yingyan Celine},
  booktitle={Forty-second International Conference on Machine Learning},
  year = {2025}
}
```
