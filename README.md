# Birds Classification

CUB-200 birds classification using PyTorch.

## Dataset

Caltech-UCSD Birds 200: http://www.vision.caltech.edu/visipedia/CUB-200.html

Download these and put them in directory `./CUB-200`

* [images.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)
* [lists.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz)

Extract the archives:

```sh
cd CUB-200
tar -xzf images.tgz
tar -xzf lists.tgz
find . -name ".*" | xargs rm -rf
```

## Environment

* Python 3.8.3
* PyTorch 1.6.0
* CUDA Version: 10.1

## Training

```sh
python main.py
```

