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

## Data

## Procedure
  
## License

This project is licensed under the terms of the MIT license.

## Publication

Mohsen Mesgar, Simon Tompson, and Iryna Gurevych. 2021. Improving Factual Consistency Between a Response and Persona Facts. In Proceedings of EACL'21. 

## BibTeX
```
@Inproceedings{mesgar21,
    title = "Neural Tree Indexers for Text Understanding",
    author = "Mesgar, Mohsen and Timpson, Edwin and Gurevych, Iryna",
    booktitle = "Proceedings of the 16th Conference of the {E}uropean Chapter of the Association for Computational Linguistics (EACL): Volume 1, Long Papers",
    month = "April",
    year = 2021,
    address = "Online",
    url = "[To Appear]",
    pages = "[To Appear]"
}
```
