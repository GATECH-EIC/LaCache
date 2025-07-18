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

## on PPL

### Installation

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
