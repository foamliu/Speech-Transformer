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
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|游客日照海森店被打受伤警方成言与出突引发服务<br>如客日照海森店被打受伤警方成言与出突引发服务<br>游客日照海身店被打受伤警方成言与出突引发服务<br>如客日照海身店被打受伤警方成言与出突引发服务<br>游客日照海深店被打受伤警方成言与出突引发服务|游客日照海鲜店被打受伤警方称言语冲突引发互殴|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|我们应该建设可控的规范化的地方政府融资机制<br>我们应该建设可空的规范化的地方政府融资机制<br>我们应该建设可控的规范划的地方政府融资机制<br>我们因该建设可控的规范化的地方政府融资机制<br>我们应该建设可控地规范化的地方政府融资机制|我们应该建设可控的规范化的地方政府融资机制|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|在布去六个月中处于高点<br>在布局六个月中处于高点<br>在布去六个月中处于高点的<br>在布局六个月中处于高点的<br>在不去六个月中处于高点|在过去六个月中处于高点|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|中国队教练图还是做出了让朱婷继续修战的这责<br>中国队教练图还是做出了让朱同继续修战的这责<br>中国队教练图还是做出了让朱婷继续休战的这责<br>中国队教练图还是做出了让朱停继续修战的这责<br>中国队教练图还是做出了让朱体继续修战的这责|中国队教练组还是做出了让朱婷继续休战的抉择|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|一计不会在短期内迅速回暖<br>一及不会在短期内迅速回暖<br>一举不会在短期内迅速回暖<br>预计不会在短期内迅速回暖<br>预及不会在短期内迅速回暖|预计不会在短期内迅速回暖|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|男友却仍然批腿偷吃<br>男友却人然批腿偷吃<br>难友却仍然批腿偷吃<br>男友却仍然批腿透吃<br>难友却人然批腿偷吃|男友却仍然劈腿偷吃|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|兰州楼市出现明显的区域分化<br>兰州楼市出现明显的区域分<br>兰州楼市出现明显的区于分化<br>兰州楼市出现明显的区域分华<br>兰州楼市出现明显地区域分化|兰州楼市出现明显的区域分化|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|这一系列公积金门槛放地额度提高的调整<br>这一系列攻积金门槛放地额度提高的调整<br>这一系列公积金门槛放低额度提高的调整<br>这一系列攻积金门槛放低额度提高的调整<br>这一系列共积金门槛放地额度提高的调整|这一系列公积金门槛放低额度提高的调整|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|十年二三岁的杨助军在北京抢劫杀害了一名出租车司机<br>十年二三岁的杨助君在北京抢劫杀害了一名出租车司机<br>十年二三岁的杨柱军在北京抢劫杀害了一名出租车司机<br>十年二三岁的杨柱君在北京抢劫杀害了一名出租车司机<br>十年二三岁的扬助军在北京抢劫杀害了一名出租车司机|时年二三岁的杨柱军在北京抢劫杀害了一名出租车司机|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|在确保系统顺利运行的情况下<br>在确保系统顺力运行的情况下<br>在确保系统顺立运行的情况下<br>在确保系统顺利运型的情况下<br>在确保系统顺丽运行的情况下|在确保系统顺利运行的情况下|