
# Cross-Modality Diffusion and Decorrelation for Speech Recognition

The code is released for Cross-Modality Diffusion and Decorrelation for Speech Recognition


## Installation
Environment setup
```sh
pip install -r requirement.txt
```
## Data preparation
Download datasets:

1. LibriSpeech ASR corpus (https://www.openslr.org/12)
2. Common voice (zh-TW) (https://commonvoice.mozilla.org/zh-TW/datasets)



# Run Codes

## Feature Extraction for Speech Representation

To extract the WavLM features with the extract.py script

```sh
python -m wavlm.extract --librispeech_path # dataset location --out_path $ output directory --ckpt_path # directory of WavLM model 
```



## Dataset Preprocessing

1. English (LibriSpeech ASR corpus )

```sh
python split_data.py --librispeech_path # directory of LibriSpeech --ls_wavlm_path # directory of extracted WavLM --include_test
```
2. Chinese (Common voice (zh-TW) )

Here, The directoryies in first.py and second.py are needed to be changed.
```sh
cd preprocessing
python first.py
python second.py
```

## Diffusion Training
Training the diffusion model with our proposed cross-modality decorreltation objective with train.py

```sh
deepspeed --include localhost:0 train.py train_csv=# training data csv valid_csv=# validation data csv  checkpoint_path # directory of checkpoint folder vocab_path=# diectory of vocab.pt  batch_size=# batch size  --deepspeed --deepspeed_config=deepspeed_cfg.json validation_interval=20000 checkpoint_interval=1000
```
## Evaluation

Evaluating the diffusion model in terms of CER, WER and time.

```sh
python -m evaluation.score --ckpt # checkpoint path --eval_csv # csv file path --vocab # vocabulary path --fs # 1 or 2 --T # total time steps
```

`--fs 1` means **Rescaled Parameterization**
`--fs 2` means **Stepwise Parameterization**
Using `--fs 1 --T 200` to run without fast sampling

Adding `--clamp` to use **Maximum Posterior**
Without adding `--clamp` to use **Average Posterior**
