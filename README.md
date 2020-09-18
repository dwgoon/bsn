# BSNet

Official Repository for "[Deep Audio Steganalysis in Time Domain](https://dl.acm.org/doi/abs/10.1145/3369412.3395064)"

## Usage

- You can automate training and testing models on a single machine.
- Edit the configuration file (*.yml) according to your environment.

```bash
python automate.py --config config_automate.yml
```

## Dependencies
- [pytorch](https://pytorch.org)
- [scikit-learn](https://scikit-learn.org)
- [numpy](https://pandas.pydata.org)
- [pandas](https://pandas.pydata.org)
- [librosa](https://github.com/librosa/librosa)
- [pyyaml](https://pyyaml.org)
- [tqdm](https://github.com/tqdm/tqdm)


## Dataset
- Cover and stego datasets were created based on [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1).
- Refer to the original paper for creating datasets.


## Citation
Cite the [original paper](https://dl.acm.org/doi/abs/10.1145/3369412.3395064) if you utilize this code.
