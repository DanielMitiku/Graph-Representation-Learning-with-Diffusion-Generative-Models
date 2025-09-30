# [Graph Representation Learning with Diffusion Generative Models](https://arxiv.org/abs/2501.13133)

### This is the official code for the paper "[Graph Representation Learning with Diffusion Generative Models](https://arxiv.org/abs/2501.13133)"

To train and evaluate the model:

```bash
python -m venv ddgae
source ddgae/bin/activate
pip install -r requirements.txt
```

Then, you can train the model from the scratch or use the pre-trained model to directly evaluate (shown in the following). To train the model from the scratch, you can:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="PROTEINS" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --epochs=200 --eval_only=0 1>train.log 2>train.err
```

To evaluate the model:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="PROTEINS" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --eval_only=1 1>eval.log 2>eval.err
```

Make sure the model is saved in the './models' directory. To use another dataset, you can change the dataset name in the args. For example, to train the IMDB-BINARY dataset, you can:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="IMDB-BINARY" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --epochs=200 --eval_only=0 1>train.log 2>train.err
```

To evaluate the IMDB-BINARY dataset, you can:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="IMDB-BINARY" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --eval_only=1 1>eval.log 2>eval.err
```

We followed similar evaluation script as in the Directional Diffusion Models [DDM](https://github.com/NeXAIS/DDM). We also thank the authors of DiffusionAE [DiffusionAE](https://github.com/phizaz/diffae) for their code.

If you find this code useful, please cite our paper from Neurips 2025 Workshop on New Perspectives in Graph Machine Learning:

```bibtex
@inproceedings{wesego2025graph,
    title={Graph Representation Learning with Diffusion Generative Models},
    author={Daniel Wesego},
    booktitle={New Perspectives in Graph Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=ZbSlY7Tc3R}
}
```