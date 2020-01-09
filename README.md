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
|Speech Transformer|11.5|[Link](https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/BEST_checkpoint.tar)|

## Dependency

- Python 3.6.8
- PyTorch 1.3.0

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
Please download the [pretrained model](https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/speech-transformer-cn.pt) then run:
```bash
$ python demo.py
```

It picks 10 random test examples and recognize them like these:

|Audio|Out|GT|
|---|---|---|
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|我国的经济处在爬破过凯的重要公考<br>我国的经济处在爬破过凯的重要公口<br>我国的经济处在盘破过凯的重要公考<br>我国的经济处在爬破过凯的重要公靠<br>我国的经济处在爬坡过凯的重要公考|我国的经济处在爬坡过坎的重要关口|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|完善主地承包经一全流市市场<br>完善主地承包经一全六市市场<br>完善主地承包经营全流市市场<br>完善主地承包经一权流市市场<br>完善主地承包经营全六市市场|完善土地承包经营权流转市场|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|临长各类设施使用年限<br>严长各类设施使用年限<br>延长各类设施使用年限<br>很长各类设施使用年限<br>难长各类设施使用年限|延长各类设施使用年限|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|苹果此举是为了节约用电量<br>苹果此举是是了节约用电量<br>苹果此举是为了解约用电量<br>苹果此举是为了节约用电令<br>苹果此举只为了节约用电量|苹果此举是为了节约用电量|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|反他们也可以有机会参与体育运动<br>让他们也可以有机会参与体育运动<br>反她们也可以有机会参与体育运动<br>范他们也可以有机会参与体育运动<br>但他们也可以有机会参与体育运动|让他们也可以有机会参与体育运动|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|陈言希穿着粉色上衣<br>陈闫希穿着粉色上衣<br>陈延希穿着粉色上衣<br>陈言琪穿着粉色上衣<br>陈演希穿着粉色上衣|陈妍希穿着粉色上衣|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|说起自己的伴女大下<br>说起自己的伴理大下<br>说起自己的半女大下<br>说起自己的办女大下<br>说起自己的半理大下|说起自己的伴侣大侠|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|每日经济新闻记者注意到<br>每日经济新闻记者朱意到<br>每日经济新闻记者注一到<br>每日经济新闻记者注注到<br>每日经济新闻记者注以到|每日经济新闻记者注意到|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|这是今年五月份以来库存环比增幅幅小了一次<br>这是今年五月份以来库存环比增幅最小了一次<br>这是今年五月份以来库存环比增幅幅小的一次<br>这是今年五月份以来库存环比增幅最小的一次<br>这是今年五月份以来库存环比增幅幅小小一次|这是今年五月份以来库存环比增幅最小的一次|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|一个人的精使生命就将走向摔老<br>一个连的精使生命就将走向摔老<br>一个人的金使生命就将走向摔老<br>一个人的坚使生命就将走向摔老<br>一个连的金使生命就将走向摔老|一个人的精神生命就将走向衰老|

## 小小的赞助~
<p align="center">
	<img src="https://github.com/foamliu/Speech-Transformer/blob/master/sponsor.jpg" alt="Sample"  width="324" height="504">
	<p align="center">
		<em>若对您有帮助可给予小小的赞助~</em>
	</p>
</p>
<br/><br/><br/>