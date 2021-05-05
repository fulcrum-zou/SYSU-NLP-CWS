# SYSU-NLP-CWS

An assignment from SYSU NLP course. A simple CWS task using LSTM-CRF.

# Environment

* CUDA

* PyTorch

# Files

* \<code\>

* \<result\>

  Training and testing output will be saved under this folder, along with `model.pkl` and test dataset segmentation result `output.utf8`.
  
* \<dataset\>


# Getting Started

Download the pre-trained word embeddings from [Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors). The embedding used in my experiment is the 300d Context Word Vectors Word -> Character (1) co-occurrence statistics.

Change training configurations in `configuration.py`

To train the model, run

```
python train.py
```

To test the model, run

```
python test.py
```





