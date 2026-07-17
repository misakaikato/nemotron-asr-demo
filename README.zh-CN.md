# Nemotron 3.5 ASR Streaming — 本地网页 Demo

[English](README.md) | **简体中文**

[nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) 的本地网页 demo,基于 [MLX](https://github.com/ml-explore/mlx) 跑在 Apple Silicon 上,完全离线。麦克风**真流式**实时字幕 + 音频文件批式转写,并自实现了 **热词偏置** 能力。

- 模型:600M 参数 cache-aware FastConformer-RNNT,单 checkpoint 覆盖 40 种语言(含中文),带标点和大小写
- 权重:[mlx-community bf16 转换版](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b)(约 1.2GB),首次启动自动下载
- M5 Max 实测:批式 RTFx 15–20,流式 RTFx 4–7(含热词偏置),中英文可用

## 环境要求

- Apple Silicon Mac(M1 及以上)—— MLX 只支持 Apple Silicon
- [uv](https://docs.astral.sh/uv/getting-started/installation/)(`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- 现代浏览器(Chrome / Safari / Edge),麦克风采集需要

## 安装与运行

```bash
git clone https://github.com/misakaikato/nemotron-asr-demo.git
cd nemotron-asr-demo
uv run server.py
```

就这些。首次 `uv run` 会自动建虚拟环境并按 `uv.lock` 装齐依赖(注意:`mlx-audio` 装的是 GitHub main 分支,Nemotron 支持尚未发布到 PyPI);首次启动会从 Hugging Face 拉取权重(约 1.2GB,之后走缓存)。

浏览器打开 **http://localhost:7860**,按提示授权麦克风。

想完全离线/自包含,可以把权重放进项目内——存在本地副本时服务端优先使用:

```bash
hf download mlx-community/nemotron-3.5-asr-streaming-0.6b \
  --local-dir models/nemotron-3.5-asr-streaming-0.6b
```

### 部署说明

- 服务默认绑定 `127.0.0.1:7860`(见 `server.py` 最后一行)。要让局域网其他设备访问,把 host 改成 `0.0.0.0`——但浏览器只在 `localhost` 或 HTTPS 源上开放麦克风,远程客户端要用麦克风需要加 TLS 反代(如 Caddy);文件转写走普通 HTTP 即可。
- 单人 demo,无鉴权、无并发控制,不要原样暴露到公网。

## 使用

**实时字幕** —— 选语言(或保持自动检测)、选延迟档,点*开始录音*直接说话。文字实时上屏,状态栏显示 音频时长 / 计算耗时 / RTFx(已剔除等麦克风的时间,>1 即快于实时)。

- **延迟档** = 模型训练过的 look-ahead(`att_context`)档位:1.12s / 0.56s / 0.32s / 0.08s。越小上屏越快,准确率略降。
- **热词** —— 每行一个词或短语,录音**开始前**填好。强度默认 3 即可;调太高(>6)会把发音相近的普通词硬掰成热词。greedy 解码无回退,词表保持在几十个以内。
- **文件转写** —— 底部上传 wav/flac/mp3 测纯速度;`samples/` 有两段合成测试音频。

命令行验证热词效果(`samples/test-zh.wav` 里的"上证指数"基线会被错听成"上政指数"):

```bash
curl -X POST "http://127.0.0.1:7860/transcribe?language=auto" \
  -F "file=@samples/test-zh.wav" -F "hotwords=上证指数" -F "hotword_weight=3"
```

### API

| 接口 | 协议 | 说明 |
|---|---|---|
| `/ws` | WebSocket | 首条消息为 JSON `{language, latency, hotwords, hotword_weight}`;之后以二进制帧发送 16kHz 单声道 int16 PCM,发文本 `stop` 结束。服务端持续推送累计结果 `{type:"partial", text, audio_s, compute_s, rtfx}`,最后推 `{type:"final"}`。 |
| `/transcribe` | POST multipart | 字段:`file`(音频)、`hotwords`(换行分隔,可选)、`hotword_weight`(可选,默认 3);查询参数 `language`(默认 `auto`)。返回 `{text, audio_s, compute_s, rtfx}`。 |

## 实现要点

- **真流式,非反复重转**:复用 mlx-audio 内部管线——`log_mel_spectrogram_frames` 增量提 mel(与整段音频逐帧等价)→ `stream_encode_chunks`(逐层 attention/conv cache,O(n) 不重算)→ 贪心 RNNT 逐帧解码。麦克风 PCM 走 WebSocket,每积累约 0.32s 的 mel 帧喂一次编码器。
- **热词偏置**(mlx-audio 没有,本项目自实现):文本空间浅融合。mlx-audio 的 tokenizer 只有 decode 没有 encode,所以不对热词分词,而是按热词的每个字符位置预扫一遍词表,建出"哪些 token 能把匹配推进几个字"的表;解码每步给这些 token 的 logit 加 `强度×推进字数`,发射后推进或淘汰匹配状态。与模型切词方式无关——中文、英文、多词短语都支持。NVIDIA 官方路线(NeMo GPU-PB / [TurboBias](https://arxiv.org/abs/2508.07014))只能跑在 CUDA 栈上。
- `server.py` 里的 `decode_biased` 复刻了 `model._decode_prompted_chunks` 的贪心循环并注入偏置——升级 mlx-audio 后若上游该函数有变,需要同步对齐这份副本。

## 项目结构

```
server.py     # FastAPI:/ws 流式、/transcribe 批式、HotwordBiaser + decode_biased
index.html    # 单页前端:麦克风采集(ScriptProcessor→16k PCM)、热词框、指标显示
samples/      # test-en.wav / test-zh.wav(macOS say 合成,验证用)
models/       # (已 gitignore)可选的本地权重,见「安装与运行」
```

## 已知限制

- ScriptProcessor 已废弃但零依赖能用,卡顿再换 AudioWorklet
- 热词不持久化(刷新即空),中途修改不影响进行中的会话
- 单人 demo:每条 WebSocket 会话一个解码线程,无排队机制

## 许可

Demo 代码:MIT。模型权重由 NVIDIA 以 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) 发布(mlx-community 转换版继承该许可),商用前请自行确认条款。
