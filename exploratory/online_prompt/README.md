# Exploration: Online prompt mining


## Description 

<br>

### Algorithm

1. Given a question $q_i$ from the train/val or test set of a QA dataset, select the groundtruth passage $p_i$.

2. Then, perform sentence tokenization with     `nltk.tokenize.sent_tokenize` on the paired passage to obtain a list of sentences $s^i_j$.

3. Find the top-k candidates determined by the cosine similarity scores of question and sentences $\textrm{similarity\_score} = \textrm{embed}(q_i) \cdot \textrm{embed}(s^i_j)$. 

4. Test the ranked candidates against the ground-truth sentence.


## Instruction:

<br>

1. Download 3 QA datasets from the following scripts

```
cd ./scripts
bash ./download_mlqa.sh
bash ./download_xorqa.en.sh
bash ./download_xquad_v1.1.sh
```

2. The `./notebooks` subdirectory contains 2 Jupyter notebooks as follows.

- `1.online_prompt_mining.compute.ipynb` for computing question-sentence pairs cosine similarity where sentences are from the groundtruth document tokenized with `nltk.tokenize.sent_tokenize`
- `2.online_prompt_mining.evaluate.ipynb` for evaluating the precision @k of the sentence mining results.

---

The data and results can be accessed via (this link)[https://vistec-my.sharepoint.com/:f:/g/personal/lalital_pro_vistec_ac_th/EvFvK3bAow9IjIazCyHlgboBmw-ApGSPglQ95T9n52EJ8w?e=sjsHG8] (currently as a private SharePoint directory)