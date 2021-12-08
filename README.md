



##  Variational Filtering

### Experiments in main paper

To run phi learning on linear gaussian (Fig1a) (x)
```
python linear_gaussian_phi_learning.py
```
To run phi and theta learning on linear gaussian (Fig1b)
```
python linear_gaussian_model_learning.py
```
To run Chaotic RNN online filtering (Fig2a)
```
python CRNN_filtering.py --config = something
```
To run Chaotic RNN comparison with offline objective (Fig2b)
```
python CRNN_filtering.py --config = something else
```
To run sequential VAE experiment (Fig3)
```
python seqVAE.py
```

### Experiments in appendix

To run amortized linear gaussian model learning
```
python linear_gaussian_model_learning_amortized.py
```
To run Chaotic RNN amortized filtering
```
python CRNN_amortized.py
```
To run amortized sequential VAE
```
python seqVAE_amortized.py
```



# Dependencies
    - pytorch
    - hydra (for hyperparameter config) https://github.com/facebookresearch/hydra
    - scipy
    - tqdm
    - matplotlib
    - tensorboard
    
# Sequential VAE demo video
https://user-images.githubusercontent.com/85300820/120666806-ca147e80-c484-11eb-8237-5dfc5083de47.mp4
