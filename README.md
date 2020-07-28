# StarGAN-Voice-Conversion


This is a pytorch implementation of the paper: [StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks](https://arxiv.org/abs/1806.02169).

**The converted voice examples are in *converted directory*.

**VCTK database has been used to train the model with 70 speakers. The convereted samples are a bit noisy because of VCTK data but it can be improved if other clean databases are used.


## [Dependencies]
- Python 3.5+
- pytorch 0.4.0+
- librosa 
- pyworld 
- tensorboardX
- scikit-learn
- tqdm


## [Usage]

### Download dataset

Download and unzip [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) corpus to designated directories.

```bash
mkdir ./data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ./data
```
If the downloaded VCTK is in tar.gz, run this:

```bash
tar -xzvf VCTK-Corpus.tar.gz -C ./data
```
The data directory now looks like this:

```
data
├── vctk
│   ├── p225
│   ├── p226
│   ├── ...
│   └── p360

```

### Preprocess

Extract features (mcep, f0, ap) from each speech clip.  The features are stored as npy files. We also calculate the statistical characteristics for each speaker.

```
python preprocess.py
```

This process may take minutes !

The data directory now looks like this:

```
data
├── vctk (48kHz data)
│   ├── p225
│   ├── p226
│   ├── ...
│   └── p360 
├── vctk_16 (16kHz data)
│   ├── p225
│   ├── p226
│   ├── ...
│   └── p360
├── mc
│   ├── train
│   ├── test


```

### Train

```
python main.py
```


### Convert


```
convert.py --src_spk p262 --trg_spk p272 --resume_iters 210000
```


## [Network structure]

![Snip20181102_2](https://github.com/hujinsen/StarGAN-Voice-Conversion/raw/master/imgs/Snip20181102_2.png)


 Note: 
 * This implementation follows the [original StarGAN-VC paper’s](https://arxiv.org/abs/1806.02169) network structure, while [StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion) use [StarGAN](https://github.com/yunjey/stargan)'s network architecture.
 * Our converted sound qualities are better than [StarGAN-VC code](https://github.com/hujinsen/pytorch-StarGAN-VC) which uses [original StarGAN-VC paper’s](https://arxiv.org/abs/1806.02169) network structure.

## [Reference]

* [StarGAN-VC paper](https://arxiv.org/abs/1806.02169)

* [StarGAN paper](https://arxiv.org/abs/1711.09020)

* [CycleGAN-VC paper](https://arxiv.org/abs/1711.11293)


## [Acknowlegements]

[StarGAN-VC code](https://github.com/hujinsen/pytorch-StarGAN-VC) (Original Network Architecture)

[StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN code](https://github.com/yunjey/stargan)

