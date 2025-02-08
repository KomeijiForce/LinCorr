# Linear Correlation in LM’s Compositional Generalization and Hallucination

How do LMs compose knowledge for generalization? Our works provide a linear correlation viewpoint on this question.

![ResilientLinearity](https://github.com/user-attachments/assets/ca222d8a-842f-48cc-8c43-59ca9777c009)

## ⚡Fast Start

We provide an example notebook ```FastLinearity.ipynb``` to fit the correlation matrix between the next token logits predicted from prompts ```X live in the city of``` and ``` X lives in the country of```. To explain the process, we sample 1000 Xs (For time efficiency, we use 10000 in our paper) from the LM's (```llama3-8b```) vocabulary and fit a linear regression between the logits.
