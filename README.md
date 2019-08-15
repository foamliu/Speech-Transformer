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
|Speech Transformer|14.9|[Link]()|

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

![image](https://github.com/foamliu/Speech-Transformer/raw/master/images/learning_rate.jpg)

![image](https://github.com/foamliu/Speech-Transformer/raw/master/images/train_loss.jpg)

![image](https://github.com/foamliu/Speech-Transformer/raw/master/images/valid_loss.jpg)

### Demo
Pick 10 random test examples from test set:
```bash
$ python demo.py
```
|Audio|Out|GT|
|---|---|---|
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|冰雪属立公共交通六千发债理念<br>冰雪属立公共交通六千发展理念<br>必雪属立公共交通六千发债理念<br>必雪属立公共交通六千发展理念<br>冰雪属立公共交通六线发债理念|必须树立公共交通优先发展理念|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|成为圈字里手去一直的超大企业<br>成为圈子里手去一直的超大企业<br>成为圈字里手去一纸的超大企业<br>成为圈子里手去一纸的超大企业<br>成为圈字里首去一直的超大企业|成为圈子里首屈一指的超大企业|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|再次降低公积金贷款首付比例<br>再次较低公积金贷款首付比例<br>再次降低公积因贷款首付比例<br>再再降低公积金贷款首付比例<br>再次价低公积金贷款首付比例|再次降低公积金贷款首付比例|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|多了出为人们的幸福观彩<br>多了初为人们的幸福观彩<br>多了出为人们的幸福光彩<br>多是出为人们的幸福观彩<br>多着出为人们的幸福观彩|多了初为人母的幸福光彩|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|但是里免不是淡顶的<br>但是里面不是淡顶的<br>但是里免不是淡定的<br>但是里面不是淡定的<br>但是里免不是半顶的|但是里面不是淡定的|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|温证火国先审后续涉案者父亲协公开到前行<br>温证火国先审后续涉案者父亲写公开到前行<br>温州火国先审后续涉案者父亲协公开到前行<br>温州火国先审后续涉案者父亲写公开到前行<br>温证火国先审后续涉案者父亲些公开到前行|温州火锅先生后续涉案者父亲写公开道歉信|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|以加强产销衔接为重点<br>一加强产销衔接为重点<br>以加强产销行接为重点<br>以家强产销衔接为重点<br>一加强产销行接为重点|以加强产销衔接为重点|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|香港海关严查水后各<br>香港海关严查水后个<br>香港海关严查水货各<br>香港海关严查水或各<br>相港海关严查水后各|香港海关严查水货客|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|背后涉及到百万车主的信息安全<br>背后涉及高百万车主的信息安全<br>背后涉及告百万车主的信息安全<br>背后涉集到百万车主的信息安全<br>背后涉级到百万车主的信息安全|背后涉及到百万车主的信息安全|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|这些税费均无法避免<br>这些税费均有法避免<br>这些税费居无法避免<br>这些税费均已法避免<br>这些税费拘无法避免|这些税费均无法避免|