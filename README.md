<h1 align="center">HittER</h1>
<h5 align="center">Hierarchical Transformers for Knowledge Graph Embeddings</h5>

HittER generates embeddings for knowledge graphs and performs link prediction using a hierarchical Transformer model.
It will appear in EMNLP 2021 ([arXiv version](https://arxiv.org/abs/2008.12813)).

## Installation

The repo requires python>=3.7, anaconda and a new env is recommended.


``` sh
conda create -n hitter python=3.7 -y # optional
conda activate hitter # optional
git clone git@github.com:microsoft/HittER.git
cd HittER
pip install -e .
```

### Data

First download the standard benchmark datasets using the commands below. Thanks [LibKGE](https://github.com/uma-pi1/kge) for providing the preprocessing scripts and hosting the data.

``` sh
cd data
sh download_standard.sh
```

### Training

Configurations for the experiments are in the `/config` folder.

``` sh
python -m kge start config/trmeh-fb15k237-best.yaml
```

The training process uses DataParallel in all visible GPUs by default, which can be overrode by appending `--job.device cpu` to the command above.

### Evaluation

You can evaluate the trained models on dev/test set using the following commands.

``` sh
python -m kge eval <saved_dir>
python -m kge test <saved_dir>
```

Pretrained [models](https://github.com/microsoft/HittER/releases/tag/v1.0.0) are also released for reproducibility.

### HittER-BERT QA experiments

QA experiment-related data can be downloaded from the release.

``` sh
git submodule update --init
cd transformers
pip install -e .
```

Run experiments
``` sh
> python hitter-bert.py --help

usage: hitter-bert.py [-h] [--dataset {fbqa,webqsp}] [--filtered] [--hitter]
                      [--seed SEED]
                      [exp_name]

positional arguments:
  exp_name              Name of the experiment

optional arguments:
  -h, --help            show this help message and exit
  --dataset {fbqa,webqsp}
                        fbqa or webqsp
  --filtered            Filtered or not
  --hitter              Use pretrained HittER or not
  --seed SEED           Seed number
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Citation

```
@inproceedings{chen-etal-2021-hitter,
    title = "HittER: Hierarchical Transformers for Knowledge Graph Embeddings",
    author = "Chen, Sanxing and Liu, Xiaodong and Gao, Jianfeng and Jiao, Jian and Zhang, Ruofei and Ji, Yangfeng",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```