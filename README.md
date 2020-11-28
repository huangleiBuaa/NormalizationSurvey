# Survey on Normalization Techniques



This repo is for our paper survey paper on normalization techniques in training deep neural networks:

**Normalization Techniques in Training DNNs: Methodology, Analysis and Application**, 

Lei Huang, Jie Qin, Yi Zhou, Fan Zhu, Li Liu and Ling Shao. 

[arXiv preprint arXiv:2009.12836](https://arxiv.org/abs/2009.12836)



We hope this repo  provide a more friendly  way for readers to review/follow the related works. 



# Table of content

-----

[TOC]



-----

## 1. Methodology

### 1.1 Normalizing Activations by Population Statistics

- Efficient BackProp. Neural Networks: Tricks of the Trade, 1998.  [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf).

- Accelerated Gradient Descent by Factor-Centering Decomposition.  Technical Report, 1998.  [paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.2167&rep=rep1&type=pdf) .

- Deep Boltzmann Machines and the Centering Trick.  Neural Networks: Tricks of the trade, 2012.  [paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.5813&rep=rep1&type=pdf).

- Deep learning made easier by linear transformations in perceptrons.  AISTATS, 2012.  [paper](http://yann.lecun.com/exdb/publis/pdf/raiko-aistats-12.pdf) .

- Mean-normalized stochastic gradient for large-scale deep learning.  ICASSP, 2014.  [paper](https://ieeexplore.ieee.org/document/6853582).

- Natural Neural Networks.  NeurIPS, 2015.  [paper](https://arxiv.org/abs/1507.00210).

- Learning Deep Architectures via Generalized Whitened Neural Networks.  ICML, 2017.  [paper](http://proceedings.mlr.press/v70/luo17a.html) 

  

### 1.2 Normalizing Activations as Functions

- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.  ICML, 2015.  [paper](https://arxiv.org/abs/1502.03167) ,  [code]().
- Knowledge matters: Importance of prior information for optimization. JMLR, 2016.  [paper](https://arxiv.org/abs/1301.4083) .
- Recurrent Batch Normalization. ICLR, 2017.  [paper](https://arxiv.org/abs/1603.09025) ,  [code](https://github.com/cooijmanstim/recurrent-batch-normalization).
- Batch normalized recurrent neural networks. ICASSP, 2016.  [paper](https://arxiv.org/abs/1510.01378).

#### 1.2.1 Normalization Area Partitioning

- Layer Normalization. arXiv:1607.06450, 2016.  [paper](https://arxiv.org/abs/1607.06450) .
- Group Normalization. ECCV, 2018.  [paper](https://arxiv.org/abs/1803.08494) ,  [code](https://github.com/ppwwyyxx/GroupNorm-reproduce).
- Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv:1607.08022, 2016.  [paper](https://arxiv.org/abs/1607.08022) ,  [code](https://github.com/DmitryUlyanov/texture_nets).
- Positional Normalization. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1907.04312) ,  [code](https://github.com/Boyiliee/PONO).
- Four Things Everyone Should Know to Improve Batch Normalization. ICLR, 2020.  [paper](https://arxiv.org/abs/1906.03548) ,  [code](https://github.com/ceciliaresearch/four_things_batch_norm).
- Local Context Normalization: Revisiting Local Normalization. CVPR, 2020.  [paper](https://arxiv.org/abs/1912.05845) ,  [code](https://github.com/anthonymlortiz/lcn).
- What is the best multi-stage architecture for object recognition?. ICCV, 2009.  [paper](https://ieeexplore.ieee.org/document/5459469) .
- ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS, 2012.  [paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) .
- Normalizing the Normalizers: Comparing and Extending Network Normalization Schemes. ICLR, 2017.  [paper](https://arxiv.org/abs/1611.04520) ,  [code](https://github.com/renmengye/div-norm).

#### 1.2.2 Normalization Operation

- Decorrelated Batch Normalization. CVPR, 2018.  [paper](https://arxiv.org/abs/1804.08450) ,  [code](https://github.com/princeton-vl/DecorrelatedBN).
- Iterative Normalization: Beyond Standardization towards Efficient Whitening. CVPR, 2019.  [paper](https://arxiv.org/abs/1904.03441) ,  [code](https://github.com/huangleiBuaa/IterNorm).
- Whitening and Coloring transform for GANs. ICLR, 2019.  [paper](https://arxiv.org/abs/1806.00420) ,  [code](https://github.com/AliaksandrSiarohin/wc-gan).
- An Investigation into the Stochasticity of Batch Whitening. CVPR, 2020.  [paper](https://arxiv.org/abs/2003.12327) ,  [code](https://github.com/huangleiBuaa/StochasticityBW).
- Network Deconvolution. ICLR, 2020.  [paper](https://arxiv.org/abs/1905.11926) ,  [code](https://github.com/yechengxi/deconvolution).
- Channel Equilibrium Networks for Learning Deep Representation. ICML, 2020.  [paper](https://arxiv.org/abs/2003.00214) ,  [code](https://github.com/Tangshitao/CENet).
- Concept Whitening for Interpretable Image Recognition.  arXiv:2002.01650, 2020.  [paper](https://arxiv.org/abs/2002.01650) ,  [code](https://github.com/zhiCHEN96/ConceptWhitening).
- IsoBN: Fine-Tuning BERT with Isotropic Batch Normalization. arXiv:2005.02178, 2020.  [paper](https://arxiv.org/abs/2005.02178).
- Streaming Normalization: Towards Simpler and More Biologically-plausible Normalizations for Online and Recurrent Learning. arXiv:1610.06160, 2016.  [paper](https://arxiv.org/abs/1610.06160).
- L1-Norm Batch Normalization for Efficient Training of Deep Neural Networks. arXiv:1802.09769,  2018.  [paper](https://arxiv.org/abs/1802.09769) .
- Norm matters: efficient and accurate normalization schemes in deep networks. NeurIPS, 2018.  [paper](https://arxiv.org/abs/1803.01814) ,  [code](https://github.com/eladhoffer/norm_matters).
- Generalized Batch Normalization: Towards Accelerating Deep Neural Networks. AAAI, 2019.  [paper](https://arxiv.org/abs/1812.03271) .
- Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. NeurIPS, 2016.  [paper](https://arxiv.org/abs/1602.07868) ,  [code](https://github.com/openai/weightnorm).
- Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization. ICLR, 2020.  [paper](https://arxiv.org/abs/2001.06838) ,  [code](https://github.com/megvii-model/MABN).
- PowerNorm: Rethinking Batch Normalization in Transformers. ICML, 2020.  [paper](https://arxiv.org/abs/2003.07845) ,  [code](https://github.com/sIncerass/powernorm).
- Progressive Growing of GANs for Improved Quality, Stability, and Variation. ICLR, 2018.  [paper](https://arxiv.org/abs/1710.10196) ,  [code](https://github.com/tkarras/progressive_growing_of_gans).
- Root Mean Square Layer Normalization. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1910.07467) ,  [code](https://github.com/bzhangGo/rmsnorm).
- Online Normalization for Training Neural Networks. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1905.05894) ,  [code](https://github.com/Cerebras/online-normalization).
- Correct Normalization Matters: Understanding the Effect of Normalization On Deep Neural Network Models For Click-Through Rate Prediction. arXiv:2006.12753, 2020.  [paper](https://arxiv.org/abs/2006.12753).

#### 1.2.3 Normalization Representation Recovery

- Whitening and Coloring transform for GANs. ICLR, 2019.  [paper](https://arxiv.org/abs/1806.00420) ,  [code](https://github.com/AliaksandrSiarohin/wc-gan).

- Dynamic Layer Normalization for Adaptive Neural Acoustic Modeling in Speech Recognition. INTERSPEECH, 2017.  [paper](https://arxiv.org/abs/1707.06065).

- Multimodal Unsupervised Image-to-Image Translation. ECCV, 2018.  [paper](https://arxiv.org/abs/1804.04732) ,  [code](https://github.com/NVlabs/MUNIT).

- U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. ICLR, 2020.  [paper](https://arxiv.org/abs/1907.10830) ,  [code](https://github.com/taki0112/UGATIT).

- Instance-Level Meta Normalization. CVPR, 2019.  [paper](https://arxiv.org/abs/1904.03516) ,  [code](https://github.com/Gasoonjia/ILM-Norm).

- Semantic Image Synthesis with Spatially-Adaptive Normalization. CVPR, 2018.  [paper](https://arxiv.org/abs/1903.07291) ,  [code](https://nvlabs.github.io/SPADE/).

- Instance Enhancement Batch Normalization: an Adaptive Regulator of Batch Noise. AAAI, 2020.  [paper](https://arxiv.org/abs/1908.04008) ,  [code](https://github.com/gbup-group/IEBN).

- Attentive Normalization. ECCV, 2020.  [paper](https://arxiv.org/abs/1908.01259) ,  [code](https://github.com/iVMCL/AOGNet-v2).

- Understanding and Improving layer normalization. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1911.07013) ,  [code](https://github.com/lancopku/AdaNorm).

- Modulating early visual processing by language. NeurIPS, 2017.  [paper](https://arxiv.org/abs/1707.00683) ,  [code](https://github.com/ap229997/Conditional-Batch-Norm).

- A Learned Representation For Artistic Style. ICLR, 2017.  [paper](https://arxiv.org/abs/1610.07629) ,  [code](https://github.com/joelmoniz/gogh-figure).

  

#### 1.2.4 Multi-Mode

- Training Faster by Separating Modes of Variation in Batch-normalized Models. TPAMI, 2019.  [paper](https://arxiv.org/abs/1806.02892) ,  [code]().
- Mode Normalization. ICLR, 2019.  [paper](https://arxiv.org/abs/1810.05466) ,  [code](https://github.com/ldeecke/mn-torch).

#### 1.2.5 Combinational Normalization

- Differentiable learning-to-normalize via switchable normalization.  ICLR, 2019.   [paper](https://arxiv.org/abs/1806.10779) ,  [code](https://github.com/switchablenorms/Switchable-Normalization).
- SSN: Learning Sparse Switchable Normalization via SparsestMax. CVPR, 2019.  [paper](https://arxiv.org/abs/1903.03793) ,  [code](https://github.com/switchablenorms/Sparse_SwitchNorm).
- Switchable Whitening for Deep Representation Learning. ICCV, 2019.  [paper](https://arxiv.org/abs/1904.09739) ,  [code](https://github.com/XingangPan/Switchable-Whitening).
- Exemplar Normalization for Learning Deep Representation. CVPR, 2020.  [paper](https://arxiv.org/abs/2003.08761).
- Differentiable Dynamic Normalization for Learning Deep Representation. ICML, 2019.  [paper](http://proceedings.mlr.press/v97/luo19a.html) .
- Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks. NeurIPS, 2018.  [paper](https://arxiv.org/abs/1805.07925) ,  [code](https://github.com/hyeonseobnam/Batch-Instance-Normalization).
- U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. ICLR, 2020.  [paper](https://arxiv.org/abs/1907.10830) ,  [code](https://github.com/taki0112/UGATIT).
- TaskNorm: Rethinking Batch Normalization for Meta-Learning. ICML, 2020.  [paper](https://arxiv.org/abs/2003.03284) ,  [code](https://github.com/cambridge-mlg/cnaps).
- Rethinking Normalization and Elimination Singularity in Neural Networks. arXiv:1911.09738, 2019.  [paper](https://arxiv.org/abs/1911.09738) ,  [code](https://github.com/joe-siyuan-qiao/Batch-Channel-Normalization).
- Evolving Normalization-Activation Layers. arXiv:2004.02967, 2020.  [paper](https://arxiv.org/abs/2004.02967) ,  [code](https://github.com/lonePatient/EvoNorms_PyTorch).

#### 1.2.6 BN for More Robust Estimation

- Kalman Normalization: Normalizing Internal Representations Across Network Layers. NeurIPS, 2018.  [paper](https://papers.nips.cc/paper/2018/file/e369853df766fa44e1ed0ff613f563bd-Paper.pdf) ,  [code](https://github.com/wanggrun/Kalman-Normalization).

##### 1.2.6.1 Normalization as Functions Combining Population Statistics

- Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models. NeurIPS, 2017.  [paper](https://arxiv.org/abs/1702.03275) .
- Density estimation using Real NVP. ICLR, 2017.  [paper](https://arxiv.org/abs/1605.08803) ,  [code](https://github.com/chrischute/real-nvp).
- Convergence Analysis of Batch Normalization for Deep Neural Nets. arXiv:1705.08011, 2017.  [paper](https://arxiv.org/abs/1705.08011v1) .
- Revisit Batch Normalization: New Understanding and Refinement via Composition Optimization. AISTATS, 2019.  [paper](http://proceedings.mlr.press/v89/lian19a.html) .
- Online Normalization for Training Neural Networks. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1905.05894) ,  [code](https://github.com/Cerebras/online-normalization).
- Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization. ICLR, 2020.  [paper](https://arxiv.org/abs/2001.06838) ,  [code](https://github.com/megvii-model/MABN).
- PowerNorm: Rethinking Batch Normalization in Transformers. ICML, 2020.  [paper](https://arxiv.org/abs/2003.07845) ,  [code](https://github.com/sIncerass/powernorm).
- Momentum Batch Normalization for Deep Learning with Small Batch Size. ECCV, 2020.  [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570222.pdf) .
- Double forward propagation for memorized batch normalization. AAAI, 2018.  [paper](https://arxiv.org/pdf/2010.04947.pdf) .
- Cross-iteration batch normalization. arXiv:2002.05712, 2020.  [paper](https://arxiv.org/abs/2002.05712) ,  [code](https://github.com/Howal/Cross-iterationBatchNorm).

##### 1.2.6.2 Robust Inference Methods for BN

- EvalNorm: Estimating Batch Normalization Statistics for Evaluation. ICCV, 2019.  [paper](https://arxiv.org/abs/1904.06031) .
- Four Things Everyone Should Know to Improve Batch Normalization. ICLR, 2020.  [paper](https://arxiv.org/abs/1906.03548) ,  [code](https://github.com/ceciliaresearch/four_things_batch_norm).
- An Investigation into the Stochasticity of Batch Whitening. CVPR, 2020.  [paper](https://arxiv.org/abs/2003.12327) ,  [code](https://github.com/huangleiBuaa/StochasticityBW).

### 1.3 Normalizing Weights

- Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. NeurIPS, 2016.  [paper](https://arxiv.org/abs/1602.07868) ,  [code](https://github.com/openai/weightnorm).

- Centered Weight Normalization  in Accelerating Training of Deep Neural Networks. ICCV, 2017.  [paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Centered_Weight_Normalization_ICCV_2017_paper.pdf) ,  [code](https://github.com/huangleiBuaa/CenteredWN).

- Orthogonal Weight Normalization: Solution to Optimization over Multiple Dependent Stiefel Manifolds in Deep Neural Networks. AAAI, 2018.  [paper](https://arxiv.org/abs/1709.06079) ,  [code](https://github.com/huangleiBuaa/OthogonalWN).

- Spectral normalization for generative adversarial networks. ICLR, 2018.  [paper](https://arxiv.org/abs/1802.05957) ,  [code](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan).

- Cosine normalization: Using cosine similarity instead of dot product in neural networks. ICANN, 2018.  [paper](https://arxiv.org/abs/1702.05870) .

- Weight standardization. arXiv:1903.10520, 2019.  [paper](https://arxiv.org/abs/1903.10520) ,  [code](https://github.com/joe-siyuan-qiao/WeightStandardization).

- Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization. ICLR, 2020.  [paper](https://arxiv.org/abs/2001.06838) ,  [code](https://github.com/megvii-model/MABN).

  #### (Approximating ) Orthogonality constraints

- Unitary Evolution Recurrent Neural Networks. ICML, 2016.  [paper](https://arxiv.org/abs/1511.06464) .

- Full-Capacity Unitary Recurrent Neural Networks. NeurIPS, 2016.  [paper](https://arxiv.org/abs/1611.00035) ,  [code](https://github.com/stwisdom/urnn).

- DizzyRNN: Reparameterizing Recurrent Neural Networks for Norm-Preserving Backpropagation. arXiv:1612.04035, 2016.  [paper](https://arxiv.org/abs/1612.04035) .

- On orthogonality and learning recurrent networks with long term dependencies. ICML, 2017.  [paper](https://arxiv.org/abs/1702.00071) ,  [code](https://github.com/veugene/spectre_release).

- Learning Unitary Operators with Help From u(n). AAAI, 2017.  [paper](https://arxiv.org/abs/1607.04903) ,  [code](https://github.com/ratschlab/uRNN).

- Gated Orthogonal Recurrent Units: On Learning to Forget. arXiv:1706.02761, 2017.  [paper](https://arxiv.org/abs/1706.02761) ,  [code](https://github.com/jingli9111/GORU-tensorflow).

- Orthogonal Weight Normalization: Solution to Optimization over Multiple Dependent Stiefel Manifolds in Deep Neural Networks. AAAI, 2018.  [paper](https://arxiv.org/abs/1709.06079) ,  [code](https://github.com/huangleiBuaa/OthogonalWN).

- Orthogonal Recurrent Neural Networks with Scaled {C}ayley Transform. ICML, 2018.  [paper](https://arxiv.org/abs/1707.09520) ,  [code](https://github.com/SpartinStuff/scoRNN).

- Fine-grained Optimization of Deep Neural Networks. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1905.09054).

- Orthogonal deep neural networks. TPAMI, 2019.  [paper](https://arxiv.org/abs/1905.05929) .

- Orthogonal Convolutional Neural Networks. CVPR, 2020.  [paper](https://arxiv.org/abs/1911.12207) ,  [code](https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks).

- Deep Isometric Learning for Visual Recognition. ICML, 2020.  [paper](https://arxiv.org/abs/2006.16992) ,  [code](https://github.com/HaozhiQi/ISONet).

- Controllable Orthogonalization in Training DNNs. CVPR, 2020.  [paper](https://arxiv.org/abs/2004.00917) ,  [code](https://github.com/huangleiBuaa/ONI).

- Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?. NeurIPS, 2018.  [paper](https://arxiv.org/abs/1810.09102) ,  [code](https://github.com/VITA-Group/Orthogonality-in-CNNs).

- Parseval Networks: Improving Robustness to Adversarial Examples. ICML, 2017.  [paper](https://arxiv.org/abs/1704.08847) ,  [code](https://github.com/Ivan1248/Parseval-networks).

- Large Scale GAN Training for High Fidelity Natural Image Synthesis. ICLR, 2019.  [paper](https://arxiv.org/abs/1809.11096) ,  [code](https://github.com/taki0112/BigGAN-Tensorflow).

- Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform. ICLR, 2020.  [paper](https://arxiv.org/abs/2002.01113) ,  [code](https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform).

### 1.4 Normalizing Gradients

- Block-normalized gradient method: An empirical study for training deep neural network. arXiv:1707.04822, 2017.  [paper](https://arxiv.org/abs/1707.04822).
- Large batch training of convolutional networks. arXiv:1708.03888, 2017.  [paper](https://arxiv.org/abs/1708.03888) ,  [code](https://github.com/noahgolmant/pytorch-lars).
- Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. ICLR, 2020.  [paper](https://arxiv.org/abs/1904.00962) ,  [code](https://github.com/ymcui/LAMB_Optimizer_TF).
- Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes. arXiv:2006.13484, 2020.  [paper](https://arxiv.org/abs/2006.13484) .
- Large Batch Training Does Not Need Warmup. arXiv:2002.01576, 2020.  [paper](https://arxiv.org/abs/2002.01576) .
- Gradient Centralization: A New Optimization Technique for Deep Neural Networks. ECCV, 2020.  [paper](https://arxiv.org/abs/2004.01461) ,  [code](https://github.com/Yonghongwei/Gradient-Centralization).

## 2 Analysis 

### 2.1 Scale Invariance in Stabilizing Training

- Layer Normalization. arXiv:1607.06450, 2016.  [paper](https://arxiv.org/abs/1607.06450).

- Data-Dependent Path Normalization in Neural Networks. ICLR, 2016.  [paper](https://arxiv.org/abs/1511.06747) .

- Riemannian approach to batch normalization. NeurIPS, 2017.  [paper](https://arxiv.org/abs/1709.09603) ,  [code](https://github.com/MinhyungCho/riemannian-batch-normalization).

- New Interpretations of Normalization Methods in Deep Learning. AAAI, 2020.  [paper](https://arxiv.org/abs/2006.09104) .

- Norm matters: efficient and accurate normalization schemes in deep networks. NeurIPS, 2018.  [paper](https://arxiv.org/abs/1803.01814) ,  [code](https://github.com/eladhoffer/norm_matters).

- Layer-wise Conditioning Analysis in Exploring the Learning Dynamics of DNNs. ECCV, 2020.  [paper](https://arxiv.org/abs/2002.10801) ,  [code](https://github.com/huangleiBuaa/LayerwiseCA).

  #### Learning Rate Auto-tuning

- Theoretical Analysis of Auto Rate-Tuning by Batch Normalization. ICLR, 2019.  [paper](https://arxiv.org/abs/1812.03981).

- Spherical Perspective on Learning with Batch Norm. arXiv:2006.13382, 2020.  [paper](https://arxiv.org/abs/2006.13382) ,  [code](https://github.com/ymontmarin/adamsrt).

- A Quantitative Analysis of the Effect of Batch Normalization on Gradient Descent. ICML, 2019.  [paper](https://arxiv.org/abs/1810.00122).

- Separating the Effects of Batch Normalization on CNN Training Speed and Stability Using Classical Adaptive Filter Theory. arXiv:2002.10674, 2020.  [paper](https://arxiv.org/abs/2002.10674) .

- L2 regularization versus batch and weight normalization. arXiv:1706.05350, 2017.  [paper](https://arxiv.org/abs/1706.05350) .

- Projection Based Weight Normalization for Deep Neural Networks. arXiv:1710.02338, 2017.  [paper](https://arxiv.org/abs/1710.02338) ,  [code](https://github.com/huangleiBuaa/NormProjection).

- Three Mechanisms of Weight Decay Regularization. ICLR, 2019.  [paper](https://arxiv.org/abs/1810.12281) ,  [code](https://github.com/gd-zhang/Weight-Decay).

- An Exponential Learning Rate Schedule For Batch Normalized Networks. ICLR, 2020.  [paper](https://arxiv.org/abs/1910.07454) .

- Spherical Motion Dynamics of Deep Neural Networks with Batch Normalization and Weight Decay. arXiv:2006.08419, 2020.  [paper](https://arxiv.org/abs/2006.08419) .

  

### 2.2 Improved Conditioning in Optimization

- Second Order Properties of Error Surfaces. NeurIPS, 1990.  [paper](http://papers.neurips.cc/paper/314-second-order-properties-of-error-surfaces-learning-time-and-generalization.pdf) .
- Decorrelated Batch Normalization. CVPR, 2018.  [paper](https://arxiv.org/abs/1804.08450) ,  [code](https://github.com/princeton-vl/DecorrelatedBN).
- How Does Batch Normalization Help Optimization?. NeurIPS, 2018.  [paper](https://arxiv.org/abs/1805.11604) .
- An Exponential Learning Rate Schedule For Batch Normalized Networks. ICLR, 2020.  [paper](https://arxiv.org/abs/1910.07454) .
- An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. ICML, 2019.  [paper](https://arxiv.org/abs/1901.10159) ,  [code](https://github.com/google/spectral-density).
- Understanding Batch Normalization. NeurIPS, 2018.   [paper](https://arxiv.org/abs/1806.02375) .
- The Normalization Method for Alleviating Pathological Sharpness in Wide Neural Networks. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1906.02926).
- Layer-wise Conditioning Analysis in Exploring the Learning Dynamics of DNNs. ECCV, 2020.  [paper](https://arxiv.org/abs/2002.10801) ,  [code](https://github.com/huangleiBuaa/LayerwiseCA).
- Theoretical Understanding of Batch-normalization: A Markov Chain Perspective. arXiv:2003.01652, 2020.  [paper](https://arxiv.org/abs/2003.01652v1).
- A Mean Field Theory of Batch Normalization. ICLR, 2019.  [paper](https://arxiv.org/abs/1902.08129) .
- Mean-field Analysis of Batch Normalization. arXiv:1903.02606, 2019.  [paper](https://arxiv.org/abs/1903.02606) .
- Characterizing Well-Behaved vs. Pathological Deep Neural Networks. ICML, 2019.  [paper](https://arxiv.org/abs/1811.03087) ,  [code](https://github.com/alabatie/moments-dnns).
- Exponential convergence rates for Batch Normalization: The power of length-direction decoupling in non-convex optimization. AISTATS, 2019.  [paper](http://proceedings.mlr.press/v89/kohler19a.html) .
- Optimization Theory for ReLU Neural Networks Trained with Normalization Layers. ICML, 2020.  [paper](https://arxiv.org/abs/2006.06878).

### 2.3 Stochasticity for Generalization



- Bayesian Uncertainty Estimation for Batch Normalized Deep Networks. ICML, 2018.  [paper](http://proceedings.mlr.press/v80/teye18a.html) .
- Stochastic Normalizations as Bayesian Learning. ACCV, 2018.  [paper](https://arxiv.org/abs/1811.00639) .
- Uncertainty Estimation via Stochastic Batch Normalization. ICLR Workshop, 2018.  [paper](https://arxiv.org/abs/1802.04893).
- Iterative Normalization: Beyond Standardization towards Efficient Whitening. CVPR, 2019.  [paper](https://arxiv.org/abs/1904.03441) ,  [code](https://github.com/huangleiBuaa/IterNorm).
- An Investigation into the Stochasticity of Batch Whitening. CVPR, 2020.  [paper](https://arxiv.org/abs/2003.12327) ,  [code](https://github.com/huangleiBuaa/StochasticityBW).
- Instance Enhancement Batch Normalization: an Adaptive Regulator of Batch Noise. AAAI, 2020.  [paper](https://arxiv.org/abs/1908.04008) ,  [code](https://github.com/gbup-group/IEBN).
- Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift. arXiv:2006.10963, 2020. [paper](https://arxiv.org/abs/2006.10963) .

## 3. Application

### 3.1 Domain Adaptation

- Revisiting Batch Normalization For Practical Domain Adaptation. arXiv:1603.04779, 2016.  [paper](https://arxiv.org/abs/1603.04779) .
- AutoDIAL: Automatic DomaIn Alignment Layers. ICCV, 2017.  [paper](https://arxiv.org/abs/1704.08082) ,  [code](https://github.com/ducksoup/autodial/blob/master/README.md).
- Domain-Specific Batch Normalization for Unsupervised Domain Adaptation. CVPR, 2019.  [paper](https://arxiv.org/abs/1906.03950) ,  [code](https://github.com/wgchang/DSBN).
- A Domain Agnostic Normalization Layer for Unsupervised Adversarial Domain Adaptation. WACV, 2019.  [paper](https://arxiv.org/abs/1809.05298) ,  [code](https://github.com/RobRomijnders/dan).
- Adversarial Examples Improve Image Recognition. CVPR, 2020.  [paper](https://arxiv.org/abs/1911.09665) .
- Unsupervised Domain Adaptation Using Feature-Whitening and Consensus Loss. CVPR, 2019.  [paper](https://arxiv.org/abs/1903.03215) ,  [code](https://github.com/roysubhankar/dwt-domain-adaptation).
- Transferable Normalization: Towards Improving Transferability of Deep Neural Networks. NeurIPS, 2019.  [paper](https://papers.nips.cc/paper/2019/file/fd2c5e4680d9a01dba3aada5ece22270-Paper.pdf) ,  [code](https://github.com/thuml/TransNorm).
- Learning to Optimize Domain Specific Normalization for Domain Generalization. ECCV, 2020.  [paper](https://arxiv.org/abs/1907.04275) .

#### Learning Universal Representations

- Universal representations: The missing link between faces, text, planktons, and cat breeds. arXiv:1701.07275, 2017.  [paper](https://arxiv.org/abs/1701.07275).
- Interpolating Convolutional Neural Networks Using Batch Normalization. ECCV, 2018.  [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Gratianus_Wesley_Putra_Data_Interpolating_Convolutional_Neural_ECCV_2018_paper.pdf) .
- Efficient Multi-Domain Learning by Covariance Normalization. CVPR, 2019.  [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Efficient_Multi-Domain_Learning_by_Covariance_Normalization_CVPR_2019_paper.pdf) ,  [code](https://github.com/liyunsheng13/Efficient-Multi-Domain-Network-Learning-by-Covariance-Normalization).

### 3.2 Style Transfer

- Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv:1607.08022, 2016.  [paper](https://arxiv.org/abs/1607.08022) ,  [code](https://github.com/DmitryUlyanov/texture_nets).
- A Learned Representation For Artistic Style. ICLR, 2017.  [paper](https://arxiv.org/abs/1610.07629) ,  [code](https://github.com/joelmoniz/gogh-figure).
- Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. ICCV, 2017.  [paper](https://arxiv.org/abs/1703.06868) ,  [code](https://github.com/xunhuang1995/AdaIN-style).
- FET-GAN: Font and Effect Transfer via K-shot Adaptive Instance Normalization. AAAI, 2020.  [paper](https://ojs.aaai.org//index.php/AAAI/article/view/5535) ,  [code](https://github.com/liweileev/FET-GAN).
- Dynamic Instance Normalization for Arbitrary Style Transfer. AAAI, 2020.  [paper](https://arxiv.org/abs/1911.06953) .
- Universal style transfer via feature transforms. NeurIPS, 2017.  [paper](https://arxiv.org/abs/1705.08086) ,  [code](https://github.com/Yijunmaverick/UniversalStyleTransfer).
- Understanding Generalized Whitening and Coloring Transform for Universal Style Transfer. ICCV, 2019.  [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf) .
- Avatar-net: Multi-scale zero-shot style transfer by feature decoration. CVPR, 2018.  [paper](https://arxiv.org/abs/1805.03857) ,  [code](https://github.com/LucasSheng/avatar-net).

#### Image Translation

- Multimodal Unsupervised Image-to-Image Translation. ECCV, 2018.  [paper](https://arxiv.org/abs/1804.04732) ,  [code](https://github.com/NVlabs/MUNIT).
- Image-to-image translation via group-wise deep whitening-and-coloring transformation. CVPR, 2019.  [paper](https://arxiv.org/abs/1812.09912) .
- Unpaired Image Translation via Adaptive Convolution-based Normalization. arXiv:1911.13271, 2019.  [paper](https://arxiv.org/abs/1911.13271) .
- Region Normalization for Image Inpainting. AAAI, 2020.  [paper](https://arxiv.org/abs/1911.10375) ,  [code](https://github.com/geekyutao/RN).
- Attentive Normalization for Conditional Image Generation. CVPR, 2020.  [paper](https://arxiv.org/abs/2004.03828) ,  [code](https://github.com/Jia-Research-Lab/AttenNorm).

### 3.3 Training GANs

- Spectral normalization for generative adversarial networks. ICLR, 2018.  [paper](https://arxiv.org/abs/1802.05957) ,  [code](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan).

- Large Scale GAN Training for High Fidelity Natural Image Synthesis. ICLR, 2019.  [paper](https://arxiv.org/abs/1809.11096) ,  [code](https://github.com/taki0112/BigGAN-Tensorflow).

- Controllable Orthogonalization in Training DNNs. CVPR, 2020.  [paper](https://arxiv.org/abs/2004.00917) ,  [code](https://github.com/huangleiBuaa/ONI).

- Modulating early visual processing by language. NeurIPS, 2017.  [paper](https://arxiv.org/abs/1707.00683) ,  [code](https://github.com/ap229997/Conditional-Batch-Norm).

- A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR, 2019.  [paper](https://arxiv.org/abs/1812.04948) ,  [code](https://github.com/NVlabs/stylegan).

- On Self Modulation for Generative Adversarial Networks. ICLR, 2019.  [paper](https://arxiv.org/abs/1810.01365) ,  [code](https://github.com/google/compare_gan).

- An Empirical Study of Batch Normalization and Group Normalization in Conditional Computation. arXiv:1908.00061, 2019.  [paper](https://arxiv.org/abs/1908.00061) .

  

### 3.4 Efficient Deep Models

- Learning Efficient Convolutional Networks through Network Slimming. ICCV, 2017.  [paper](https://arxiv.org/abs/1708.06519) ,  [code](https://github.com/liuzhuang13/slimming).
- Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers. ICLR, 2018.  [paper](https://arxiv.org/abs/1802.00124) ,  [code](https://github.com/jack-willturner/batchnorm-pruning).
- EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. ECCV, 2020.  [paper](https://arxiv.org/abs/2007.02491) ,  [code](https://github.com/anonymous47823493/EagleEye).
- Slimmable Neural Networks. ICLR, 2019.  [paper](https://arxiv.org/abs/1812.08928) ,  [code](https://github.com/JiahuiYu/slimmable_networks).
- Finet: Using Fine-grained Batch Normalization to Train Light-weight Neural Networks. arXiv:2005.06828, 2020.  [paper](https://arxiv.org/abs/2005.06828) .
- Scalable methods for 8-bit training of neural networks. NeurIPS, 2018.  [paper](https://arxiv.org/abs/1805.11046) ,  [code](https://github.com/eladhoffer/quantized.pytorch).
- Low-precision batch-normalized activations. arXiv:1702.08231, 2017.  [paper](https://arxiv.org/abs/1702.08231).
- Optimal Quantization for Batch Normalization in Neural Network Deployments and Beyond. arXiv:2008.13128, 2020.  [paper](https://arxiv.org/abs/2008.13128) .
- Learning Recurrent Binary/Ternary Weights. ICLR, 2019.  [paper](https://arxiv.org/abs/1809.11086) ,  [code](https://github.com/arashardakani/Learning-Recurrent-Binary-Ternary-Weights).
- Normalization Helps Training of Quantized LSTM. NeurIPS, 2019.  [paper](https://proceedings.neurips.cc/paper/2019/hash/f8eb278a8bce873ef365b45e939da38a-Abstract.html) ,  [code](https://github.com/houlu369/Normalized-Quantized-LSTM).
- How Does Batch Normalization Help Binary Training. arXiv:1909.09139, 2019.  [paper](https://arxiv.org/abs/1909.09139) .

### 3.5 Meta learning

- On first-order meta-learning algorithms. arXiv:1803.02999, 2018.  [paper](https://arxiv.org/abs/1803.02999) ,  [code](https://github.com/openai/supervised-reptile).
- Meta-Learning Probabilistic Inference for Prediction. ICLR, 2019.  [paper](https://arxiv.org/abs/1805.09921) ,  [code](https://github.com/Gordonjo/versa).
- TaskNorm: Rethinking Batch Normalization for Meta-Learning. ICML, 2020.  [paper](https://arxiv.org/abs/2003.03284) ,  [code](https://github.com/cambridge-mlg/cnaps).

### 3.6 Reinforcement learning

- Learning values across many orders of magnitude. NeurIPS, 2016.  [paper](https://arxiv.org/abs/1602.07714) ,  [code](https://github.com/brendanator/atari-rl/blob/master/README.md).
- Crossnorm: Normalization for off-policy td reinforcement learning. arXiv:1902.05605, 2019.  [paper](https://arxiv.org/abs/1902.05605).
- Striving for Simplicity and Performance in Off-Policy DRL: Output Normalization and Non-Uniform Sampling. ICML, 2020.  [paper](https://arxiv.org/abs/1910.02208) ,  [code](https://github.com/Fable67/Streamlined-Off-Policy-Learning).

### 3.7 Unsupervised/Semi-supervised representation learning

- Momentum Contrast for Unsupervised Visual Representation Learning. CVPR, 2020.  [paper](https://arxiv.org/abs/1911.05722) ,  [code](https://github.com/facebookresearch/moco).
- Unsupervised Batch Normalization. CVPR Workshops, 2020.  [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Kocyigit_Unsupervised_Batch_Normalization_CVPRW_2020_paper.pdf).
- Exploring Simple Siamese Representation Learning. arXiv:2011.10566, 2020.  [paper](https://arxiv.org/pdf/2011.10566.pdf).

### 3.8 Applied in miscellaneous networks.

- Learning to find good correspondences. CVPR, 2018.  [paper](https://arxiv.org/abs/1711.05971) ,  [code](https://github.com/vcg-uvic/learned-correspondence-release).
- Attentive context normalization for robust permutation-equivariant learning. CVPR, 2020.  [paper](https://arxiv.org/abs/1907.02545) ,  [code](https://github.com/vcg-uvic/acne).
- GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training. arXiv:2009.03294, 2020.  [paper](https://arxiv.org/abs/2009.03294) ,  [code](https://github.com/lsj2408/GraphNorm).
- Towards Understanding Normalization in Neural ODEs. arXiv:2004.09222, 2020.  [paper](https://arxiv.org/abs/2004.09222) .
- Riemannian batch normalization for SPD neural networks. NeurIPS, 2019.  [paper](https://arxiv.org/abs/1909.02414) .
- Batch Normalization is a Cause of Adversarial Vulnerability. arXiv:1905.02161, 2019.  [paper](https://arxiv.org/abs/1905.02161) .
- Towards an Adversarially Robust Normalization Approach. arXiv:2006.11007, 2020.  [paper](https://arxiv.org/abs/2006.11007) ,  [code](https://github.com/awaisrauf/RobustNorm).
- Intriguing Properties of Adversarial Training at Scale. ICLR, 2020.  [paper](https://arxiv.org/abs/1906.03787) ,  [code](https://github.com/tingxueronghua/pytorch-classification-advprop).



-----

## Contact

* **Lei Huang** - huanglei36060520 [at] gmail.com