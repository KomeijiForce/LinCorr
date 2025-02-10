# Linear Correlation in LM’s Compositional Generalization and Hallucination

How do LMs compose knowledge for generalization? Our work provides a linear correlation viewpoint on this question. Link to the paper: [https://arxiv.org/abs/2502.04520](https://arxiv.org/abs/2502.04520)

![ResilientLinearity](https://github.com/user-attachments/assets/ca222d8a-842f-48cc-8c43-59ca9777c009)

## ⚡Fast Start

### FastLinearity.ipynb

We provide an example notebook ```FastLinearity.ipynb``` to fit the correlation matrix between the next token logits predicted from prompts ```X live in the city of``` and ``` X lives in the country of```. To explain the process, we sample 1000 Xs (For time efficiency, we use 10000 in our paper) from the LM's (```llama3-8b```) vocabulary and fit a linear regression between the logits.

By checking the weights inside the fitted weights, we can find a reflection of real-word knowledge composition

```python
Y1_name = " Tokyo"
idx = Y1_names.index(Y1_name)
top_jds = weights[idx].argsort()[::-1][:5]
top_Y2_names = [Y2_names[jdx] for jdx in top_jds]
print(Y1_name, top_Y2_names)
```

which outputs ```Tokyo [' Japan', ' Luxembourg', ' Netherlands', ' Belgium', ' Nederland']```. You might get a slightly different result because only 1000 samples are used for fitting. You can set the sample number higher like 10000 to get a more precise and stable result, which will also increase the running time.

### FastResilience.ipynb

We provide an example notebook ```FastLinearity.ipynb``` to show the linear regression fitted by the logits before the large-scale post-training well adapts to the LM after post-training. We use the evaluation result from the ```correlation``` function to demonstrate a resilient linearity against fine-tuning.


## 🐾 Citation

```
@article{DBLP:journals/corr/abs-2502-04520,
  author       = {Letian Peng and
                  Chenyang An and
                  Shibo Hao and
                  Chengyu Dong and
                  Jingbo Shang},
  title        = {Linear Correlation in {LM}'s Compositional Generalization and Hallucination},
  journal      = {CoRR},
  volume       = {abs/2502.04520},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2502.04520},
  doi          = {10.48550/arXiv.2502.04520},
  eprinttype   = {arXiv},
  eprint       = {2502.04520},
  timestamp    = {Mon, 10 Feb 2025 15:07:27 -0800},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2502-04520.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
