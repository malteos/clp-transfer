# CLP-Transfer: Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning

We introduce a cross-lingual and progressive transfer learning approach, called CLP-Transfer, that transfers models from a source language, for which pretrained models are publicly available, like English, to a new target language. As opposed to prior work, which focused on the cross-lingual transfer between two languages, we extend the transfer to the model size. Given a pretrained model in a source language, we aim for a same-sized model in a target language. Instead of training a model from scratch, we exploit a smaller model that is in the target language but requires much fewer resources. Both small and source models are then used to initialize the token embeddings of the larger model based on the overlapping vocabulary of the source and target language. All remaining weights are reused from the model in the source language. This approach outperforms the sole cross-lingual transfer and can save up to 80% of the training steps compared to the random initialization. 

- **Preprint:** https://arxiv.org/abs/2301.09626
- **Demo:** [European Language Grid](https://live.european-language-grid.eu/catalogue/tool-service/20825/try%20out/)

<img alt="Tokens vs PPL" src="https://github.com/malteos/clp-transfer/raw/main/german-6b-ppl.png">


## Pretrained models

| **Model**         | **Parameters** |
|-------------------|---------------:|
| [`bloom-6b4-clp-german`](https://huggingface.co/malteos/bloom-6b4-clp-german)`  |   6.4B |  
| [`bloom-1b5-clp-german`](https://huggingface.co/malteos/bloom-1b5-clp-german)`  |   1.5B |  
| [`gpt2-xl-wechsel-german`](https://huggingface.co/malteos/gpt2-xl-wechsel-german)`  |   1.5B |  
| [`gpt2-xl-german-covid-19`](https://huggingface.co/malteos/gpt2-xl-german-covid-19)`  |   1.5B |  


## Usage

To apply CLP-Transfer, you need a large source model (e.g., in English) and a small model in your target language.

```bash
# helper: other model in target language but with same tokenizer (smaller or other architecture)
# source: same size as target model but different language/multilingual
python clp.py apply_clp \
    --source_model_name_or_path bloom-7b1 \
    --helper_model_name_or_path gpt2-xl-wechsel-german \
    --target_model_path <output_dir>
```

## How to cite

If you are using our code or models, please cite [our paper](https://arxiv.org/abs/2301.09626):

```bibtex
@misc{Ostendorff2023clp,
  doi = {10.48550/ARXIV.2301.09626},
  author = {Ostendorff, Malte and Rehm, Georg},
  title = {Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning},
  publisher = {arXiv},
  year = {2023}
}

```

## License

Code: MIT

Pretrained models: See Huggingface