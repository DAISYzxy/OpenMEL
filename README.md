<div align="center">
<h1>OpenMEL: Unsupervised Multimodal Entity Linking Using Noise-Free Expanded Queries and Global Coherence</h1>

[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/DAISYzxy/OpenMEL)
[![GitHub Issues](https://img.shields.io/github/issues/DAISYzxy/OpenMEL.svg)](https://github.com/DAISYzxy/OpenMEL/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/DAISYzxy/OpenMEL.svg)](https://github.com/DAISYzxy/OpenMEL/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/DAISYzxy/OpenMEL.svg)](https://github.com/DAISYzxy/OpenMEL/stargazers)
[![GitHub license](https://img.shields.io/github/license/DAISYzxy/OpenMEL.svg)](https://github.com/DAISYzxy/OpenMEL/blob/main/LICENSE)
[![HitCount](https://views.whatilearened.today/views/github/DAISYzxy/OpenMEL.svg)](https://github.com/DAISYzxy/OpenMEL)
</div>

<img src="fig/framework.png" width="1000px">


**📖New Dataset Published!📖** We identify and process a new dataset, WeiboNewsMEL, which is a subset derived from Weibo-MEL. Weibo-MEL is not used in the previous works since it contains many low-quality blog contents. Accordingly, we filter the news-related instances from Weibo-MEL, excluding the lower-quality blogs to ensure a reliable evaluation. The WeiboNewsMEL comprises 1,499 textual instances accompanied by 1,288 images. We still utilize Wikidata as the knowledge base. Additionally, we open-source the WeiboNewsMEL dataset ([Google Drive link](https://drive.google.com/drive/folders/1dhQyPwOe3UJn1LHYKGvnXpzZ0YWF5eI1?usp=sharing)) to facilitate further research in this area.


# Contents

- [Introduction](#Introduction)
- [Installation](#Installation)
- [Test](#Test)
- [Evaluation](#Evaluation)

# Introduction

We propose a novel unsupervised learning framework, OpenMEL, for solving the MEL task. We enhance the textual modality contextual information by incorporating full context comprehension and general knowledge, and generates three levels of visual inputs for further adaptive selection to handle noise. To capture global entity coherence, we construct a tree cover structure, defining it as a maximum spanning tree with bounded nodes to meet the MEL objective. We then introduce a greedy algorithm with theoretical guarantees to solve this problem.



# Installation
To install the cutting edge version of `OpenMEL` from the main branch of this repo, run:
```bash
git clone https://github.com/DAISYzxy/OpenMEL.git
cd OpenMEL
pip install -r requirements.txt
```
To download the test data, open the [Google Drive link](https://drive.google.com/drive/folders/1dhQyPwOe3UJn1LHYKGvnXpzZ0YWF5eI1?usp=sharing).


# Test
Test the performance of OpenMEL, download the test data and place it under the OpenMEL folder. Then run to test WikiMEL, if you want to test other datasets, please remeber to change the file path name in main.py:
```bash
python main.py
```


# Evaluation
We present the comprehensive evaluation results in the figure below, including those on the processed WeiboNewsMEL dataset.

<img src="fig/full_results.png" width="1000px">
