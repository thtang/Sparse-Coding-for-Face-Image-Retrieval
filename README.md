# Sparse-Coding-for-Face-Image-Retrieval

## Data
LFW DATA Labeled Faces in the Wild (LFW) is a widely used benchmark for face verification. It contains 13233 images of 5749 different people. You can find more information on the data set at http://vis-www.cs.umass.edu/lfw/.
## Usage
```
cd ./src
python3 face_image_retrieval.py [LFW pickle data] [LFW attribute txt]
```
## Dependency
`Python3` `numpy` `joblib` `scipy` `pandas` `matplotlib` `spams`
## Baseline result
[Experiment report](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/report.pdf)

Compute L2 distance of local binary pattern **(LBP)**  and rank.

Query |*1st*    |  *2nd* | *3rd* |*4th* |*5th*
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/query.png)|![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/1.png)  |  ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/2.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/3.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/4.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/5.png)
![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/query_2.png)|![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/2_1.png)  |  ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/2_2.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/2_3.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/2_4.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/2_5.png)
<img src="https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/t1.png" width="620">

## Sparse coding approach
Compute Cosine similarity of **Spare coding** with identity information  and rank.

Query |*1st*    |  *2nd* | *3rd* |*4th* |*5th*
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/query.png)|![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/3_1.png)  |![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/3_2.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/3_3.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/3_4.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/3_5.png)
![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/query_2.png)|![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/4_1.png)  |  ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/4_2.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/4_3.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/4_4.png) | ![](https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/4_5.png)

* mAP of vanilla sparse coding
<img src="https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/t3.png" width="620">

* mAP of sparse coding with identity information
<img src="https://github.com/thtang/Sparse-Coding-for-Face-Image-Retrieval/blob/master/images/t5.png" width="620">

## Reference
Scalable Face Image Retrieval using Attribute-Enhanced Sparse Codewords, *Chen et al., IEEE Trans. Multimedia 2013* [[1]](http://cmlab.csie.ntu.edu.tw/~sirius42/papers/tmm12.pdf)

Semi-supervised face image retrieval using sparse coding with identity constraint, *Chen et al., ACM MM 2011* [[2]](http://cmlab.csie.ntu.edu.tw/~sirius42/papers/mm11.pdf)

SPArse Modeling Software [[3]](http://spams-devel.gforge.inria.fr/index.html)
