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
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|天然气用户为优先允许限制类和禁止量内<br>天然气用户为优先允许限制类和禁止质内<br>天然气用户为优先允许限制类和禁止量量<br>天然气用户为优先允许限制类和禁止量类<br>天然气用户为优先允许限制类和禁止质量|天然气用户为优先允许限制类和禁止类|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|从一月的一线城市的土地成交盘<br>从一月的一线城市的土地成交看<br>从一月的一线城市的土地成交态<br>从一月的一线城市的土地成交判<br>从一月的一线城市的土地成交谈|从一月的一线城市的土地成交看|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|电话手表的宿射主要来自天线<br>电话手表的负射主要来自天线<br>电话手表的复射主要来自天线<br>电话手表的扶射主要来自天线<br>电话手表的副射主要来自天线|电话手表的辐射主要来自天线|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|以区域为中心建立客户经理负责制制造方面<br>已区域为中心建立客户经理负责制制造方面<br>以区域为中心建立客户经理负责至制造方面<br>以区域为中心建立客户经理负责致制造方面<br>已区域为中心建立客户经理负责至制造方面|以区域为中心建立客户经理负责制制造方面|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|成非集成相关人士告诉每日经济性无记者<br>城非集成相关人士告诉每日经济性无记者<br>成非集成相关人士告诉每日经济性目记者<br>成非集成相关人士告诉每日经济性吴记者<br>城非集成相关人士告诉每日经济性目记者|成飞集成相关人士告诉每日经济新闻记者|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|朱婷已一百五十七分领先群放<br>朱婷以一百五十七分领先群放<br>朱婷已一百五十七分领先群方<br>朱婷已一百五十七分领先群报<br>朱婷以一百五十七分领先群方|朱婷以一百五十七分领先群芳|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|优化生产布局和续续群结构<br>优化生产布局和续续权结构<br>优化生产布局和续区群结构<br>优化生产布局和续据群结构<br>优化生产布局和续需群结构|优化生产布局和畜群结构|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|还有很多阻碍要解决<br>还有很多阻碍了解决<br>还有很多阻碍药解决<br>还油很多阻碍要解决<br>还有很多阻爱要解决|还有很多阻碍要解决|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|许如以科地发微博感谢婚姻带来的幸福<br>许如以特地发微博感谢婚姻带来的幸福<br>徐如以科地发微博感谢婚姻带来的幸福<br>许如营科地发微博感谢婚姻带来的幸福<br>许如于科地发微博感谢婚姻带来的幸福|许茹芸特地发微博感谢婚姻带来的幸福|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|门头沟验池镇村民李东梅因不服行政批复<br>门头沟验试镇村民李东梅因不服行政批复<br>门头沟验视镇村民李东梅因不服行政批复<br>门头沟验吃镇村民李东梅因不服行政批复<br>门头沟验池镇村民李东没因不服行政批复|门头沟雁翅镇村民李冬梅因不服行政批复|