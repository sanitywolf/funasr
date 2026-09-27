# funasr: 基于大模型的批量语音识别和翻译

> 用python实现的批量语音转文字和译成中文

---

## 简介 | Introduction

funasr是一组python写的工具，会加载大模型实现批量的语音转文字，及翻译成中文，输出srt字幕文件和txt文件。

它使用 CTranslate2引擎 faster_whisper_{small,midium,large-v3}三个模型  用于多语种语音识别。

它使用 facebook/nllb-200-distilled-600M  用于多语种到中文的翻译。

它使用 CPU运行大模型，需至少4G，推荐8G以上内存，可以用在GPU配置低，或没有GPU的电脑上。

它第一次运行时会自动从 hf-mirror.com 下载大模型到本地，之后就不再下载。也可以事先手动下载。

这些大模型是可以免费使用的。

---

## 各种工具

**extract_audio.py** 
```bash
(funasr_clean) D:\funars>python extract_audio.py
```

它会把放在v目录下的影音文件中的音频处理成同名的wav文件，放到v2t目录下。

先要分析源文件的声道组成，如是5.1声道，则取ch2_FC_中置人声这个声道输出标准 【16kHz单声道16bit 小端序的】 WAV文件，否则取所有声道合并为一个标准 【16kHz单声道16bit 小端序的 】WAV文件。

语音转文字大模型‌最偏好的是「16kHz 采样率、单声道、16bit 小端序的原始PCM音频」‌，这是几乎所有主流ASR大模型训练时的标准输入格式，能直接喂给模型，零额外转码损耗，识别准确率和处理速度都能拉满。

但是，pcm格式的音频没有文件头说明，使用时需额外指定参数，不方便，所以采用了wav格式。

**extract_audio-other.py**
```bash
(funasr_clean) D:\funars>python extract_audio-other.py
```
这个相似，但它不做声道分析，直接将所有声道合并成一个同名的m4a文件。

**funasr.py**
```bash
(funasr_clean) D:\funasr>python funasr.py help
```
📖 使用说明：

  模型可选：small（默认） / medium / large

  语言可选：en / zh / ja / ko（不传则自动检测每个音频的语言）

  参数顺序任意，不需要记位置


示例：
  python funasr.py                # 默认small模型，逐文件自动检测语言

  python funasr.py medium         # medium模型，逐文件自动检测语言

  python funasr.py en             # small模型，所有文件按英语识别

  python funasr.py large ja       # large模型，所有文件按日语识别


它会把v2t目录下的wav当成中文语音处理，生成同名的_{语种名}.txt 和_{语种名}.srt两个格式的文件。
不指定模型大小时，使用small；不指定语种时，根据wav文件自动分析。

它会使用CTranslate2引擎 faster_whisper_{small,midium,large-v3}三个模型之一进行语音识别，这个模型是多语种的，支持en,ja,ko,zh。


**funtranslate.py**
```bash
(funasr_clean)D:\funars>python funtranslate.py
```
它会把v2t目录下的文件，找出文件名带有*_ja.* *_ko.* *_en.*的，调用facebook/nllb-200-distilled-600M模型，译出对应的中文*_zh.*文件。


---


## 快速开始 | Quick Start

### 环境要求

- Python >= 3.11
- ffmpeg
```bash
C:\Users\xxxf>python -V
Python 3.11.9

C:\Users\xxxf>ffmpeg -version
ffmpeg version N-113238-gbb819a4ef8-20240109 Copyright (c) 2000-2024 the FFmpeg developers
built with gcc 13.2.0 (crosstool-NG 1.25.0.232_c175b21)
.....
```


### 1. 安装python虚拟环境

```bash
 mkdir d:\funars
 mkdir d:\funars\v
 mkdir d:\funars\v2t
 cd d:\funars

#将 这些文件拷贝到d:\funasr目录下：

#2026/09/22  13:49             8,824 extract_audio-other.py
#2026/09/25  15:53            10,817 extract_audio.py
#2026/09/27  13:57            14,766 funasr.py
#2026/09/27  14:33             4,744 funtranslate.py
#2026/09/27  15:46                60 requirements.txt

D:\funars>python -m venv funasr_clean

#激活虚拟环境
funasr_clean\Scripts\activate.bat

(funasr_clean) D:\funars>

#注：退出虚拟环境是funasr_clean\Scripts\deactivate.bat


#升级pip并安装PyTorch(仅CPU)

(funasr_clean) D:\funars>python -m  pip install --upgrade pip
(funasr_clean) D:\funars>pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

(funasr_clean) D:\funars>python -c "import torch; print('PyTorch 版本:', torch.__version__); print('是否可用 CUDA:', torch.cuda.is_available())"
#PyTorch 版本: 2.14.0+cpu
#是否可用 CUDA: False

#安装requirement.txt列出的依赖包，
(funasr_clean) D:\funars>pip install -r requirements.txt

```


---

## 目录结构 | Project Structure

```
./funasr
├── .cache
│   ├── huggingface
│   │   ├── hub
│   │   │   └── models--facebook--nllb-200-distilled-600M
│   │   │       └── snapshots
│   │   │           └── f8d333a098d19b4fd9a8b18f94170487ad3f821d
│   │   │               ├── config.json
│   │   │               ├── generation_config.json
│   │   │               ├── pytorch_model.bin
│   │   │               ├── README.md
│   │   │               ├── sentencepiece.bpe.model
│   │   │               ├── special_tokens_map.json
│   │   │               ├── tokenizer.json
│   │   │               └── tokenizer_config.json
│   │   └── xet
│   └── whisper
│       ├── large-v3
│       │   ├── config.json
│       │   ├── model.bin
│       │   ├── preprocessor_config.json
│       │   ├── README.md
│       │   ├── tokenizer.json
│       │   └── vocabulary.json
│       ├── medium
│       │   ├── config.json
│       │   ├── model.bin
│       │   ├── README.md
│       │   ├── tokenizer.json
│       │   └── vocabulary.txt
│       └── small
│           ├── config.json
│           ├── model.bin
│           ├── README.md
│           ├── tokenizer.json
│           └── vocabulary.txt
├── .gitignore
├── extract_audio.py
├── extract_audio-other.py
├── funasr.py
├── funtranslate.py
├── README.md
├── requirements.txt
├── 日语测试素材.zip
├── 英文会议与电话录音_测试素材.zip
└── 韩语测试素材.zip

```

---

## 数据说明

- **数据源**：所有大模型从hf-mirror.com下载，本地使用，无需注册、无限流。
- **结果可用性**：识别和翻译出的内容不会完全正确，最后还需人工校对。

---

## 许可证 | License

GPL v3 
