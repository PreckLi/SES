# SES
Official implementation of __SES: Bridging the Gap Between Explainability and Prediction of Graph Neural Networks__. Personal websites of the main authors: [Zhenhua Huang](https://zhenhuascut.github.io/), [Kunhao Li](https://preckli.github.io/)<br><br><br>
![SES](https://github.com/PreckLi/SES/blob/main/mainfig.png)
## Requirements
- torch==2.0.0  
- torch_geometric==2.3.0
## Datasets
- The real-world datasets include Cora, CiteSeer, and PolBlogs.  
- The synthetic datasets include BA-Shape, BA-Community, Tree-Cycle, and Tree-Grid.
Specifically, the download URL of datasets can refer to https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
## Run
```
  python main.py
```
## Cite
- Cite as follows:
```
@INPROCEEDINGS{10597945,
  author={Huang, Zhenhua and Li, Kunhao and Wang, Shaojie and Jia, Zhaohong and Zhu, Wentao and Mehrotra, Sharad},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)}, 
  title={SES: Bridging the Gap Between Explainability and Prediction of Graph Neural Networks}, 
  year={2024},
  volume={},
  number={},
  pages={2945-2958},
  keywords={Training;Bridges;Accuracy;Reliability engineering;Data engineering;Graph neural networks;Generators;Graph Neural Networks;Model Explanation;Node Classification;Self-Supervised Learning},
  doi={10.1109/ICDE60146.2024.00229}}
```
