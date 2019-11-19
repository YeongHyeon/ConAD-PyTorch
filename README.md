Consistency-based anomaly detection (ConAD)
=====

Implementation of Consistency-based anomaly detection (ConAD) from paper <a href="https://arxiv.org/abs/1810.13292">'Anomaly Detection With Multiple-Hypotheses Predictions'</a> with MNIST dataset [<a href="https://github.com/YeongHyeon/CVAE-AnomalyDetection">Related repository</a>].

## Architecture
<div align="center">
  <img src="./figures/conad.png" width="500">  
  <p>Simplified ConAD architecture.</p>
</div>

## Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="800">  
  <p>Graph of ConAD.</p>
</div>

## Results
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by CondAD.</p>
</div>

<div align="center">
  <img src="./figures/test-box.png" width="350"><img src="./figures/histogram-test.png" width="390">
  <p>Box plot and histogram of restoration loss in test procedure.</p>
</div>

<div align="center">
  <img src="./figures/test-latent.png" width="350">
  <p>Latent space of each class.</p>
</div>

## Environment
* Python 3.7.4  
* PyTorch 1.1.0
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Duc Tam Nguyen, et al. (2018 arXiv, 2019 ICML). <a href="https://arxiv.org/abs/1810.13292">Anomaly Detection With Multiple-Hypotheses Predictions.</a> <a href="https://icml.cc/Conferences/2019/Schedule?showEvent=4558">ICML 2019</a>.
