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
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|并未能给这些公司带来其他的效益<br>并以能给这些公司带来其他的效益<br>并未能给这些公司带来启他的效益<br>并未能给这些公司带来起他的效益<br>并未能给这些公司带来其他的效应|并未能给这些公司带来其他的效应|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|很多家长都在给台子购置各种学习用品<br>很多家长都在给孩子购置各种学习用品<br>很多家长都在给台子更置各种学习用品<br>很多家长东在给台子购置各种学习用品<br>很多家长都在给孩子更置各种学习用品|很多家长都在给孩子购置各种学习用品|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|在加上互联网企业对服务器技术性可能的要可可高<br>在加上互联网企业对服无器技术性可能的要可可高<br>在加上互联网企业对服务器技术性可能的要可很高<br>在加上互联网企业对服无器技术性可能的要可很高<br>在加上互联网企业对服务器技术性可能的要很可高|再加上互联网企业对服务器技术性可能等要求很高|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|每股报价报收于一百一十一点二七美元<br>微股报价报收于一百一十一点二七美元<br>每股报价报收馀一百一十一点二七美元<br>每股爆价报收于一百一十一点二七美元<br>每股报价暴收于一百一十一点二七美元|每股报价报收于一百一十一点二七美元|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|但各年三月却有分享一张女方做大腿的照片<br>但各年三月却又分享一张女方做大腿的照片<br>但个年三月却有分享一张女方做大腿的照片<br>但个年三月却又分享一张女方做大腿的照片<br>但各年三月却有分享一张女方做大腿开照片|但隔年三月却又分享一张女方坐他大腿的照片|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|尽环检察官最后已事时不清<br>尽环检查官最后已事时不清<br>仅环检察官最后已事时不清<br>尽环检察官最后已事实不清<br>仅环检查官最后已事时不清|尽管检察管最后以事事实不清|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|二零一五年六月七日<br>二零一五年六六七日<br>二零一五年一月七日<br>二零一五年六月七<br>二零一五年六十七日|二零一五年六月七日|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|苹果就公布的合作伙伴<br>苹果有公布的合作伙伴<br>苹果就公布了合作伙伴<br>苹果有公布了合作伙伴<br>苹果就公布的合作伙观|苹果就公布了合作伙伴|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|而南昌除了放松首套房解定标准<br>而南昌除了放松首套房界定标准<br>而南昌出了放松首套房解定标准<br>而南昌除了放松首套房借定标准<br>而南昌出了放松首套房界定标准|而南昌除了放松首套房界定标准|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|不仅受立来很好的品牌形象<br>不仅受立来良好的品牌形象<br>不仅受立来爱好的品牌形象<br>不仅受立来人好的品牌形象<br>不仅受立来联好的品牌形象|不仅树立了良好的品牌形象|