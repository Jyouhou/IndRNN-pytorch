# indRNN-pt
PyTorch implementation of [Independently Recurrent Neural Networks](https://arxiv.org/pdf/1803.04831.pdf) by Shuai Li et al. (accepted to CVPR2018).

- Official source code in [Theano and Lasagne](https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne) and [PyTorch](https://github.com/Sunnydreamrain/IndRNN_pytorch)
- [TensorFlow version](https://github.com/batzner/indrnn)
- another [PyTorch version](https://github.com/StefOe/indrnn-pytorch)

# How to Run
## test the trained model
on MNIST: `python3 Sequential_task.py --test --test_model=./best_model_mnist.pt --dataset=mnist --log_folder=./test`

## train
1. on MNIST: `python3 Sequential_task.py --dataset=mnist`
2. on pMNIST: `python3 Sequential_task.py --dataset=pmnist`

## Env

```bash
Python 3.6.6
pytorch 0.4.0
torchvision 0.2.1
cuda 8.0
numpy 1.14.5
```

# Result

| task  | valid | test |
|:------:|:------:|:------:|
| sequential-mnist |  98.98 | 98.80 |
| p-mnist | 94.22<br> (94.40 from <br>[Theano&Lasagne](https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne)) | - |
| fashion-mnist | 91.60 | - |
