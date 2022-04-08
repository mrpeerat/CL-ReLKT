# CL-ReLKT (Crosslingual-Retrieval Language Knowledge Transfer)

CL-ReLKT: Cross-lingual Language Knowledge Transfer for Multilingual Retrieval Question Answering, NAACL-2022 (Finding)

<img width="600" alt="Screen Shot 2022-04-08 at 2 06 50 PM" src="https://user-images.githubusercontent.com/21156980/162382700-a56a0d6e-e56d-4f83-80bc-dd09c72f6152.png">

## Motivation
Cross-Lingual Retrieval Question Answering (CL-ReQA) is concerned with retrieving answer documents or passages to a question written in a different language. A common approach to CL-ReQA is to create a multilingual sentence embedding space such that question-answer pairs across different languages are close to each other. 

In this paper, our goal is to improve the robustness of multilingual sentence embedding (yellow box) that works with a wide range of languages, including those with a limited amount of training data. Leveraging the generalizability of knowledge distillation, we propose a Cross-Lingual Retrieval Language Knowledge Transfer (CL-ReLKT) framework. 

### Multilingual Embedding Space Before & After performining the CL-ReLKT
<img width="645" alt="Screen Shot 2022-04-08 at 3 29 12 PM" src="https://user-images.githubusercontent.com/21156980/162397024-ee4efd19-fb74-4eba-8428-14fc3b038fe8.png">

## Paper
Link: https://openreview.net/forum?id=y42xxJ_xx8 (Not the final version)

## Citation
```
@inproceedings{limkonchotiwat-etal-2022-cl-relkt,
    title = "{CL-ReLKT}: Cross-lingual Language Knowledge Transfer for Multilingual Retrieval Question Answering",
    author = "Limkonchotiwat, Peerat  and
      Ponwitayarat, Wuttikorn  and
      Udomcharoenchaikit, Can  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    booktitle = "Findings of the North American Chapter of the Association for Computational Linguistics: NAACL 2022"
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Model and Datasets
- Model: [CLICK HERE](https://vistec-my.sharepoint.com/:f:/g/personal/peerat_l_s19_vistec_ac_th/EqOlCAwSqI1Mg6zONNoblEsBN3O2zZCfKTpBzCWhyBHv_w?e=zcuYGd)
- Datasets: [CLICK HERE](https://vistec-my.sharepoint.com/:f:/g/personal/peerat_l_s19_vistec_ac_th/EmVNratSZBZBu4sRd5CP5SQByMVkPwBPtVVyO1gCXBN2KQ?e=zePde2)
- Docker: Coming Soon

## How to train

### Step1: Triplet loss warmup step 
- Run [warmup.sh](1_use_finetune_warmup.sh)
- In this step, we finetune the mUSE model with our training data (i.e., XORQA, MLQA, or XQUAD), where the anchor is the question, the positive is the answer to the question, and the negative is obtained from bm25.

### Step2: Triplet loss online training
- Run [teacher.sh](2_use_finetune_teacher.sh)
- In this step, we continue to finetune the model in Step 1 by using triplet loss and the concept of online mining (negative mining technique). 

### Step3: Language Knowledge Transfer (Distillation)
- Run [distillation.sh](3_use_finetune_distillation.sh)
- In this step, we initialize the model's weight from Step 2 and finetune it with the language knowledge transfer technique (Section 2.2).
- We use 3 terms minimization such as question(English)-question(Non-English), document-document, document-question(non-English) as shown in the figure:

<img width="505" alt="Screen Shot 2022-04-08 at 3 26 19 PM" src="https://user-images.githubusercontent.com/21156980/162396394-6414ee8c-a43c-47aa-aeb8-041189f7d2af.png">

## Performance
<img width="700" alt="Screen Shot 2022-04-08 at 3 00 54 PM" src="https://user-images.githubusercontent.com/21156980/162392211-56dd939c-b998-4cf0-9a53-c394021fbfb4.png">
<img width="400" alt="Screen Shot 2022-04-08 at 2 58 26 PM" src="https://user-images.githubusercontent.com/21156980/162391939-67d943fc-11b1-4fec-99be-78b971329ef7.png">
