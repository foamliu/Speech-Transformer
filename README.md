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
Pick 10 random test examples from test set:
```bash
$ python demo.py
```
|Audio|Out|GT|
|---|---|---|
|[audio_0.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_0.wav)|这个的斗和考验有有就就发在我们身边变<br>这个的斗和考验有有时就发在我们身边变<br>这样的斗和考验有有就就发在我们身边变<br>这样的斗和考验有有时就发在我们身边变<br>这个的斗和考验有有就就发在我们身边|这样的斗争和考验有时就发生在我们身边|
|[audio_1.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_1.wav)|加快实施保护性更多通程<br>加快实施保护性更多工程<br>加快实施保护性更多同程<br>加快实施保护性更多空程<br>加快实施保护性更多控程|加快实施保护性耕作工程|
|[audio_2.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_2.wav)|但是由于该项业务占比较小<br>但是由于单项业务占比较小<br>但是由于该项业务占比焦小<br>但是由于该项业务占比交小<br>但是由于该项业务站比较小|但是由于该项业务占比较小|
|[audio_3.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_3.wav)|以陈马镇就是玩笑任何情况都力挺范冰冰<br>以陈马镇就是玩笑任何情况都立挺范冰冰<br>以陈马镇就是玩笑任何情况都力挺范冰丁<br>以陈马镇就是玩笑任何情况都力挺饭冰冰<br>李陈马镇就是玩笑任何情况都力挺范冰冰|李晨马震就是玩笑任何情况都力挺范冰冰|
|[audio_4.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_4.wav)|经过将近一年时间的卖场谈判<br>经过将近一年时间的卖长谈判<br>经过将近一年时间的漫场谈判<br>经过将近一年时间的迈场谈判<br>经过将近一年时间的漫长谈判|经过将近一年时间的漫长谈判|
|[audio_5.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_5.wav)|中国和多米尼家同期九分排在第四和平无<br>中国和多米尼家同期九分排在第四和零无<br>中国和多米尼家同期九分排在第四和评无<br>中国和多米尼加同期九分排在第四和平无<br>中国和多米尼家同期九分排在第四和林无|中国和多米尼加同积九分排在第四和第五位|
|[audio_6.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_6.wav)|温州一网友造谣苏迪罗登陆期间水库公他悲拘<br>温州一网友造谣苏迪罗登陆期间水库公踏悲拘<br>温州一网友造谣苏迪罗登陆期间水库公她悲拘<br>温州一网友造谣苏迪罗登陆期间水库公他被拘<br>温州一网友造谣苏迪罗登陆期间水库公踏被拘|温州一网友造谣苏迪罗登陆期间水库崩塌被拘|
|[audio_7.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_7.wav)|他看了不少恐部片<br>她看了不少恐部片<br>他看了不少恐不片<br>他看了不少恐补片<br>他看了不少恐步片|她看了不少恐怖片|
|[audio_8.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_8.wav)|对于这条特殊的岁道<br>对于这条特殊的碎道<br>对于这条特殊的最道<br>对于这条特殊的随道<br>对于这条特殊的水道|对于这条特殊的隧道|
|[audio_9.wav](https://github.com/foamliu/Speech-Transformer/raw/master/audios/audio_9.wav)|报七千亿美元<br>报期千亿美元<br>到七千亿美元<br>爆七千亿美元<br>暴七千亿美元|报七千亿美元|