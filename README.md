# iSTFTNet : Fast and Lightweight Mel-spectrogram Vocoder Incorporating Inverse Short-time Fourier Transform
This repository is based on the [opensource implementation](https://github.com/rishikksh20/iSTFTNet-pytorch) of [iSTFTNet](https://arxiv.org/abs/2203.02395) (model `C8C8I`). Our contribution to the repository:

- shared the weights of the model we trained on robust internal dataset consists of `Russian speech` recorded in different acoustic conditions with sample rate `22050 Hz`;
- added `loguru` & `wandb`; 
- added `Docerfile` for faster env set up;
- updated the code with several scripts to `compute mel-spectrograms` and `convert the model to .onnx`.

Note: according to our tests `iSTFT Net` shows even higher synthesis quality than [`HiFi GAN`](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/HiFiGAN), with a 2x acceleration of RTF.


## Table of Contents  
- [Setup env](#setup-env)  
- [Inference](#inference) 
- [Train](#train)
- [ONNX](#onnx)
- [Citations](#citations)
- [References](#references)

## Setup env

### Docker
```shell
bash run_docker.sh
```     
      
### Conda 
```shell
conda create —name istft-vocoder python=3.10
pip install -r requirements.txt
````       
      
## Inference 

### Download checkpoints

```shell
bash download_checkpoints.sh
```   
Your file structure should look like:
      
```shell
├── data                                                                                                                                                                                 
│   ├── awesome_checkpoints                                                                                                                                                              
│   │   ├── do_00975000                                                                                                                                                                  
│   │   ├── g_00975000                                                                                                                                                                   
│   │   └── g_00975000.onnx                                                                                                                                                              
│   ├── deep_voices_mel                                                                                                                                                                  
│   │   ├── andrey_preispolnilsya.npy                                                                                                                                                    
│   │   ├── egor_dora.npy
│   │   └── kirill_lunch.npy
│   └── deep_voices_wav
│       ├── andrey_preispolnilsya.wav
│       ├── egor_dora.wav
│       └── kirill_lunch.wav
```      
 
Note: we trained the model with batch size 16 using 4 a100 GPUs for ~1M steps.
 
| Filename  | Description |
| ------------- | ------------- |
|do_00975000 | Discriminator checkpoint.|
|g_00975000 | Generator checkpoint. |
| g_00975000.onnx | `.onnx` model. |
|deep_voices_mel | Directory with 3 mel-spectrograms of test-audios.|
|deep_voices_wav | Directory with 3 original audios – voices of our team, this audios were not seen during the training.|

 ### Inference 
 
To run inference with downloaded test-files:
```shell
python -m src.inference
```       
       
To run inference with your own files or parameters:

| Parameter  | Description |
| ------------- | ------------- |
| config_path | Path to [`config.json`](iSTFTNet-pytorch/config/config.json).|
| input_wavs_dir | Directory with your wav files to synthesize, default is `/app/data/deep_voices_wavs`  |
| input_mels_dir  | Directory with pre-computed mel-spectrograms to synthesize mel. Note that mel-spectrograms should be computed with [compute_mels_from_audio.py](iSTFTNet-pytorch/scripts/compute_mels_from_audio.py) script, default is `/app/data/deep_voices_mels`.|
|compute_mels| Pass `--no-compute_mels` if you precomputed mels, if not specified mels will be computed from the audios in input_wavs_dir.|
|onnx_inference| If specified, checkpoint file should be `.onnx` file.|
|onnx_provider| Used if onnx_inference is specified, default provider is `CPUExecutionProvider` for `CPU` inference.
|checkpoint_file| Path to the generator checkpoint or `.onnx` model.|
|output_dir | Path where generated wavs will be saved, default is `/app/data/generated_files`.|
     

## Train 

To train the model:
1. Login from CLI to Wanb account: `wandb login`
2. Create `train.txt` and `val.txt` with [create_manifests.py](iSTFTNet-pytorch/scripts/create_manifests.py).
3. Run `src.train`

Parameters for training and finetuning the model:

| Parameter  | Description |
| ------------- | ------------- |
| input_training_file | Path to the `train.txt`.|
| input_validation_file | Path to the `val.txt`.  |
|config_path | Path the [`config.json`](iSTFTNet-pytorch/config/config.json).|
|input_mels_dir | Path to the directory with mel-spectrograms, specify if you would like to train / finetune the model on Acoustic Model outputs. |
| fine_tuning | If specified will look for mel-spectrograms in `input_mels_dir`.|
|checkpoint_path | Path to the directory with checkpoints, if you would like to finetune the model on your data based on our checkpoints: `/app/new_checkpoints`. |
|training_epochs | `N` epochs to train the model. |
|wandb_log_interval | `N` steps through which log training loss to wandb. |
| checkpoint_interval |`N` steps through which save checkpoint. |
| log_audio_interval | `N` steps through which log generated audios from validation dataset to wandb. |
| validation_interval | `N` steps through which run validation and log validation loss to wandb. |

Note: for correct inference and finetuning from our checkpoints, parameters: `num_mels`, `n_fft`, `hop_size`, `win_size`, `sampling_rate`, `fmin` and `fmax` should not be changed. 


## ONNX

Find the instructions to infer `.onnx` model in the `Inference` block. To convert trained model to `.onnx`:
```shell
python -m srcipts.convert_to_onnx
```      
      
| Parameter  | Description |
| ------------- | ------------- |
| checkpoint_file | Path to the `generator` checkpoint.   |
| config_path | Path to the [`config.json`](iSTFTNet-pytorch/config/config.json).  |
| converted_model_path | Path where converted model will be saved, default is `/app/istft_vocoder.onnx`. |

## Citations
```
@inproceedings{kaneko2022istftnet,
title={{iSTFTNet}: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform},
author={Takuhiro Kaneko and Kou Tanaka and Hirokazu Kameoka and Shogo Seki},
booktitle={ICASSP},
year={2022},
}
```

```
@misc{deepvk2023istft,
  author = {Daria, Diatlova},
  title = {istft-net},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/deepvk/istft-net}}
}
```

## References
* https://github.com/rishikksh20/iSTFTNet-pytorch
