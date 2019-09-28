# SE_SSCNN (Scale Equivariant Scale Steerable CNN)

For Scale Steerable (Invariant) CNN code: https://github.com/rghosh92/SS-CNN

The main code is in main_test.py: Train equivariant (or invariant) networks of your choice on various datasets. 
Note that the datasets would have to be provided as a folder with the appropriate name, and as pickle files. 

There are two options for the steerable filter dictionary for creating scale-equivariant CNNs:

(i)  Log-Radial Harmonics (Net_steergroupeq_xxxx) 

(2)  2D Discrete Cosine Transform Basis (Net_steergroupeq_xxxx_dctbasis, Current SoTA on MNIST-Scale-10k)

xxxx: (Dataset Name)

There are networks for other datasetes as well (mnist, fashion-mnist and cifar-10).



