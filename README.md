# Order-Agnostic Cross Entropy (OAXE) for Non-Autoregressive Machine Translation


### Note

We based this code heavily on the original code of mask-predict and disco.


### Download model 
Description | Dataset | Model
---|---|---
CMLM-OAXE | [WMT14 English-German] | [download (.tar.bz2)](TBD)
CMLM-OAXE | [WMT14 German-English] | [download (.tar.bz2)](TBD)
CMLM-OAXE | [WMT16 English-Romanian] | [download (.tar.bz2)](TBD)
CMLM-OAXE | [WMT16 Romanian-English] | [download (.tar.bz2)](TBD)
CMLM-OAXE | [WMT17 English-Chinese] | [download (.tar.bz2)](TBD)
CMLM-OAXE | [WMT17 Chinese-English] | [download (.tar.bz2)](TBD)


### Preprocess

TBD

### Train
Here we offer the XE finetuning strategy, you have to copy a pretrained CMLM model into your model dir. We adopt the pretrain models from [Mask-Predict](https://github.com/facebookresearch/Mask-Predict). It is better to keep update-freq x GPU NUM >= 8.
```
data_dir = data dir
model_dir = model dir
python3 -u train.py ${data_dir} --arch bert_transformer_seq2seq \
    --criterion oaxe --lr 5e-6  \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self \
    --max-tokens 16384 --weight-decay 0.01 --dropout 0.2 --encoder-layers 6 --encoder-embed-dim 512 \
    --decoder-layers 6 --decoder-embed-dim 512 --max-source-positions 10000 \
    --max-target-positions 10000 --max-epoch 245 --seed 1 \
    --save-dir ${model_dir} \
    --keep-last-epochs 10 --share-all-embeddings \
    --keep-interval-updates 1 --fp16 \
    --update-freq 1 --ddp-backend=no_c10d \
    --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 500 \
    --label-smoothing 0.1 --reset-optimizer --skip 0.15 \
```

### Evaluation

TBD

# License
Following MASK-PREDICT, our code is also CC-BY-NC 4.0.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

```bibtex
@inproceedings{Du2021OAXE,
  title = {Order-Agnostic Cross Entropy for Non-Autoregressive Machine Translation},
  author = {Cunxiao Du and Zhaopeng Tu and Jing Jiang},
  booktitle = {Proc. of ICML},
  year = {2021},
}
```
