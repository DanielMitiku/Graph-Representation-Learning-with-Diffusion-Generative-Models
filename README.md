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
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="IMDB-BINARY" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --epochs=200 --eval_only=0 1>train.log 2>train.err
```

To evaluate the model:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="IMDB-BINARY" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --eval_only=1 1>eval.log 2>eval.err
```

Make sure the model is saved in the './models' directory. To use another dataset, you can change the dataset name in the args. For example, to train the PROTEINS dataset, you can:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="PROTEINS" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --epochs=200 --eval_only=0 1>train.log 2>train.err
```

To evaluate the PROTEINS dataset, you can:

```bash
python train_gnn_diffae2.py --model_type="beatsganunet" --dataset="PROTEINS" --z_dim=64 --adj_max_size=64 --use_middle_blk=1 --eval_only=1 1>eval.log 2>eval.err
```

