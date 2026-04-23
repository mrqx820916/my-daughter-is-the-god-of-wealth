# 《我的女儿是财神》

> 都市·重生·女强·逆袭·爽文 | 4卷105章 · 32.7万字 · 已完结

## 📖 一句话简介

天庭做了三百年财神的小姑娘偷溜下凡，投胎成了被弃孕妇肚里的女儿——她妈正被渣男逼着打胎。

## 🎯 核心卖点

- **双强母女**：妈刚被扫地出门，闺女转头就带妈捡漏赚一个亿
- **反差爽**：婆婆嫌女儿是赔钱货，结果闺女是行走的印钞机
- **搞钱花样多**：古玩捡漏→赌石暴富→股票封神→商战碾压→世界首富
- **天庭视角穿插**：小财神在娘胎里就有记忆，内心OS又萌又飒

## 📚 章节结构

| 卷 | 主题 | 章节 | 字数 |
|----|------|------|------|
| 卷一 | 财神降世（古玩捡漏） | V01C001-V01C020（20章） | ~6.4万字 |
| 卷二 | 石破天惊（赌石暴富） | V02C021-V02C056（36章） | ~10.8万字 |
| 卷三 | 股海翻云（投资封神） | V03C057-V03C083（27章） | ~8.8万字 |
| 卷四 | 封神之路（商业帝国） | V04C084-V04C105（22章） | ~6.8万字 |

## 👥 主要人物

| 角色 | 身份 | 配音方案 |
|------|------|----------|
| 小七/金七七 | 天庭财神下凡（女儿） | 童声（pyworld升调） |
| 林若溪 | 女主/妈妈 | 原声中文女 |
| 陆北辰 | 周氏财团继承人/男主 | 低沉男声（pyworld降调） |
| 顾明轩 | 渣男前夫 | 原声中文男 |
| 陈甜甜 | 小三/绿茶 | 原声中文女 |
| 李氏 | 尖酸婆婆 | 老年女声（pyworld降调） |
| 王妈 | 忠心保姆 | 老年女声（pyworld降调） |
| 林书远 | 女主父亲 | 原声中文男 |
| 陈思琪 | 海外投资经理 | 原声中文女 |

## 🗂 项目结构

```
xiaoshuo/
├── README.md           # 本文件（项目总览）
├── outline.md          # 章节大纲（105章详细梗概）
├── characters.md       # 人物设定（9个主要角色）
├── worldbuilding.md    # 世界观设定（天庭/财运体系）
├── style-guide.md      # 写作风格指南
├── index.html          # SPA在线阅读器（暗色主题）
├── serve.py            # Web服务器（Python HTTP, 端口8083）
├── tts_server.py       # TTS语音合成服务器（CosyVoice, 端口8084）
├── voice.html          # 角色试听页面
├── build.sh            # 构建/统计/导出脚本
├── chapters/           # 105章正文（Markdown）
│   ├── V01C001_天庭偷溜.md
│   ├── ...
│   └── V04C105_做了300年财神.md
└── audio/              # TTS合成音频输出
    ├── cosyvoice_samples/   # CosyVoice对比试听
    └── samples/             # 角色试听样本
```

## 🚀 快速开始

### 在线阅读

```bash
python3 serve.py
# 浏览器打开 http://localhost:8083
```

### 统计信息

```bash
bash build.sh           # 查看字数统计
bash build.sh export    # 导出离线HTML
```

### TTS语音合成（需要GPU）

```bash
# 前置：安装CosyVoice + 下载模型
pip install cosyvoice pyworld librosa
# 模型下载到 /root/cosyvoice_models/CosyVoice-300M-SFT/

# 启动TTS服务
python3 tts_server.py --port 8084

# 访问试听页面
# http://localhost:8083/voice.html
```

## 🎙 TTS技术方案

### 方案探索历程

| 方案 | 模型 | 结果 |
|------|------|------|
| ~~ChatTTS~~ | ChatTTS 0.1 | ❌ 音质差、OOM、男女不分 |
| ~~CosyVoice2-0.5B~~ | CosyVoice2-0.5B | ❌ 需BFloat16，GTX 1050Ti不支持 |
| ~~Instruct纯指令~~ | CosyVoice-300M-Instruct | ⚠️ 只改语气不改音色，F0几乎无变化 |
| **SFT + pyworld变调** | CosyVoice-300M-SFT + pyworld | ✅ 最终方案 |

### 最终方案：CosyVoice SFT + pyworld后处理

- **TTS引擎**：CosyVoice-300M-SFT（阿里开源，Float16，4GB显存足够）
- **基础音色**：中文女（F0≈221Hz）+ 中文男（F0≈233Hz）
- **角色区分**：pyworld基频变换
  - 童声（小七）：F0 ×1.5 → ~328Hz
  - 老年女（王妈/李氏）：F0 ×0.75 → ~164Hz
  - 低沉男（陆北辰）：F0 ×0.75 → ~177Hz
  - 年轻女性/原声男：保持不变
- **推理速度**：RTF 1.3-1.9（实时率），4秒音频约6-8秒生成
- **显存占用**：加载1666MB，推理峰值1791MB

### 硬件环境

- GPU：NVIDIA GTX 1050 Ti 4GB
- CUDA：12.2（nvidia-driver-535）
- Python：3.11（venv）
- PyTorch：2.5.1+cu121

### 当前状态

- ✅ Web阅读器完成
- ✅ TTS引擎选型完成（CosyVoice SFT + pyworld）
- ✅ 角色音色方案确定（10角色 → 2基础 + pyworld变调）
- 🔄 **进行中**：tts_server.py 集成 pyworld 后处理
- ⏳ **待完成**：V01C001 完整合成验证
- ⏳ **待完成**：105章批量音频合成

## ✍ 写作规范

- 章节文件命名：`V{卷号}C{章号}_标题.md`
- 内心OS格式：用【】包裹
- 章节标题：`#标题`（一级标题）
- 每章2300-4500字，章均3115字
- 详细风格指南见 `style-guide.md`

## 📋 目标平台

番茄免费小说

## 📄 许可

个人项目，仅供学习交流。
