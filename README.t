# Speech Transformer

## Introduction

This is a PyTorch re-implementation of Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition.

## Dataset

Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.

400 people from different accent areas in China are invited to participate in the recording, which is conducted in a quiet indoor environment using high fidelity microphone and downsampled to 16kHz. The manual transcription accuracy is above 95%, through professional speech annotation and strict quality inspection. The data is free for academic use. We hope to provide moderate amount of data for new researchers in the field of speech recognition.
```
@inproceedings{aishell_2017,
  title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
  author={Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng},
  booktitle={Oriental COCOSDA 2017},
  pages={Submitted},
  year={2017}
}
```
In data folder, download speech data and transcripts:

```bash
$ wget http://www.openslr.org/resources/33/data_aishell.tgz
```

## Performance

Evaluate with 7176 audios in Aishell test set:
```bash
$ python test.py
```

## Results

|Model|CER|Download|
|---|---|---|
|Speech Transformer|12.1|[Link](https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/BEST_checkpoint.tar)|

## Dependency

- Python 3.5.2
- PyTorch 1.0.0

## Usage
### Data Pre-processing
Extract data_aishell.tgz:
```bash
$ python extract.py
```

Extract wav files into train/dev/test folders:
```bash
$ cd data/data_aishell/wav
$ find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

Scan transcript data, generate features:
```bash
$ python pre_process.py
```

Now the folder structure under data folder is sth. like:
<pre>
data/
    data_aishell.tgz
    data_aishell/
        transcript/
            aishell_transcript_v0.8.txt
        wav/
            train/
            dev/
            test/
    aishell.pickle
</pre>

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Pick 10 random test examples from test set:
```bash
$ python demo.py
```
|Audio|Out|GT|
|---|---|---|
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|$(out_list_0)|$(gt_0)|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|$(out_list_1)|$(gt_1)|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|$(out_list_2)|$(gt_2)|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|$(out_list_3)|$(gt_3)|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|$(out_list_4)|$(gt_4)|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|$(out_list_5)|$(gt_5)|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|$(out_list_6)|$(gt_6)|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|$(out_list_7)|$(gt_7)|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|$(out_list_8)|$(gt_8)|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|$(out_list_9)|$(gt_9)|