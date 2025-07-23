# GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification

Aitao Yang; Min Li; Yao Ding; Leyuan Fang; Yaoming Cai; Yujie He

___________

The code in this toolbox implements the ["GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification"]( https://ieeexplore.ieee.org/document/10746459). 



Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

A. Yang, M. Li, Y. Ding, L. Fang, Y. Cai and Y. He, "GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-14, 2024, Art no. 5537414, doi: 10.1109/TGRS.2024.3493101.

      @ARTICLE{10746459,
      author={Yang, Aitao and Li, Min and Ding, Yao and Fang, Leyuan and Cai, Yaoming and He, Yujie},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification}, 
      year={2024},
      volume={62},
      number={},
      pages={1-14},
      keywords={Feature extraction;Transformers;Semantics;Data mining;Vectors;Hyperspectral imaging;Computational efficiency;Training;Kernel;Encoding;Graph convolutional network (GCN);hyperspectral image (HSI) classification;mamba;remote sensing;state space model (SSM)},
      doi={10.1109/TGRS.2024.3493101}}

    
System-specific notes
---------------------
The codes of networks were tested using PyTorch 2.1.1 version (CUDA 11.8) in Python 3.8 on Ubuntu system.

How to use it?
---------------------
Directly run **GraphMamba.py** functions with different network parameter settings to produce the results. Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper.

For the datasets:
Add your dataset path to function “load_dataset” in function.py

On the Indian Pines dataset, you can either re-train by following:
 `python MAIN_Mamba_ip.py`

On the Salinas dataset, you can either re-train by following:
 `python MAIN_Mamba_sa.py`

On the UH2018 dataset, you can either re-train by following:
 `python MAIN_Mamba_uh2018.py`



