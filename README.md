# Improving Factual Consistency Between a Response and Persona Facts

Neural models for response generation produce responses that are semantically plausible but not necessarily factually consistent with facts describing the speaker's persona. 
These models are trained with fully supervised learning where the objective function barely captures factual consistency. 
We propose to fine-tune these models by reinforcement learning and an efficient reward function that explicitly captures the consistency between a response and persona facts as well as semantic plausibility. Our automatic and human evaluations on the PersonaChat corpus confirm that our approach increases the rate of responses that are factually consistent with persona facts over its supervised counterpart while retains the language quality of responses.  


Your interest to this project is very appreciated. 
Please read and cite [this paper]() to optimize what you can get from this repo. 
Also please don't forget to give it a Github star (on top right).


https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Setup

```bash
conda create -n ttrl

source activate ttrl 

conda install pip

pip install -r requirements.txt
```

## Data


## Procedure

1. Persona-consistency subreward
The accuracy of our BERT-based model for this subreward is obtained by 
```bash
python persona-consistency-subreward/eval_on_dialogue_nli.py 
```   

2. Language quality subreward
This folder contains the scripts related to the language qaulity subreward. 
```bash
python finetuned_lm_sample_test.py
```

3. To train:
```bash
python train.py
```
# Running ConvAI2 evaluation scripts

To run the evaluation scripts of the ConvAI2 challenge, you first need to install `ParlAI` in the repo base folder like this:

```bash
cd ParlAI
python setup.py develop
```

You can then run the evaluation script from `ParlAI` base folder:

```bash
cd ParlAI
python ../convai_evaluation_edit.py  
```   

## License

This project is licensed under the terms of the MIT license.

## Publication

Mohsen Mesgar, Simon Simpson, and Iryna Gurevych. 2021. Improving Factual Consistency Between a Response and Persona Facts. In Proceedings of EACL'21. 

## BibTeX
```
@Inproceedings{mesgar21,
    title = "Improving Factual Consistency Between a Response and Persona Facts",
    author = "Mesgar, Mohsen and Simpson, Edwin and Gurevych, Iryna",
    booktitle = "Proceedings of the 16th Conference of the {E}uropean Chapter of the Association for Computational Linguistics (EACL): Volume 1, Long Papers",
    month = "April",
    year = 2021,
    address = "Online",
    url = "[To Appear]",
    pages = "[To Appear]"
}
```
