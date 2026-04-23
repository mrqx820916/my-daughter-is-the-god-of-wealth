#!/usr/bin/env python3
"""
小说TTS语音合成服务器 - CosyVoice3 多角色语音
用法:
  WSL2: conda activate cosyvoice && python tts_server.py --serve --port 8084
  或:   python tts_server.py --samples     # 生成角色试听
        python tts_server.py --chapter V01C001
        python tts_server.py --all
依赖: CosyVoice3 (FunAudioLLM/CosyVoice), torch, torchaudio, modelscope
"""
import os, sys, json, re, time, argparse

# CosyVoice 路径
COSYVOICE_DIR = os.environ.get(
    "COSYVOICE_DIR",
    r"D:\CosyVoice"
)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, 'third_party/Matcha-TTS'))
sys.path.insert(0, COSYVOICE_DIR)

import numpy as np

NOVEL_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(NOVEL_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    os.path.join(COSYVOICE_DIR, "pretrained_models/Fun-CosyVoice3-0.5B")
)

# 参考音频目录（存放 zero_shot 角色的参考音频）
PROMPT_DIR = os.path.join(NOVEL_DIR, "audio", "prompts")
os.makedirs(PROMPT_DIR, exist_ok=True)

# ============================================================
# 角色声音配置 - CosyVoice3 混合方案
# mode:
#   "zero_shot" - 用参考音频克隆音色（主角，效果最好）
#   "instruct2" - 用自然语言指令控制音色（配角，无需参考音频）
# ============================================================
CHARACTERS = {
    "小七": {
        "gender": "female",
        "desc": "天庭财神/女儿，童声但老灵魂",
        "sample_text": "妈咪，那个垃圾股其实是个宝贝哦~本财神当年点拨过范蠡，教过沈万三呢！",
        "mode": "instruct2",
        "ref_wav": "ref_child_girl.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个四五岁的小女孩，声音非常稚嫩、清脆、甜美、可爱，"
            "带着奶声奶气的童真感，语调活泼上扬，充满天真和快乐。"
            "<|endofprompt|>"
        ),
    },
    "林若溪": {
        "gender": "female",
        "desc": "女主/妈妈，温柔坚定",
        "sample_text": "我什么都不要，只要我的孩子。从今天起，我不会再让任何人欺负我们母女。",
        "mode": "instruct2",
        "ref_wav": "ref_female_gentle.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个二十五六岁的年轻女性，声音温柔、舒缓、柔和，"
            "带有坚定的力量感，语速适中，充满母性的温暖和关怀。"
            "<|endofprompt|>"
        ),
    },
    "陆北辰": {
        "gender": "male",
        "desc": "男主，腹黑深情商界精英",
        "sample_text": "你是我见过唯一一个让数据说谎的人。林若溪，你让我很感兴趣。",
        "mode": "instruct2",
        "ref_wav": "ref_male_deep.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个三十多岁的成熟男性，声音低沉、浑厚、富有磁性，"
            "音调偏低，语速沉稳，充满自信和掌控力，像商业精英。"
            "<|endofprompt|>"
        ),
    },
    "王妈": {
        "gender": "female",
        "desc": "忠心保姆，泼辣心善，偏老年女声",
        "sample_text": "小姐，您放心，有我王妈在，谁也别想欺负您和小姐！",
        "mode": "instruct2",
        "ref_wav": "ref_auntie.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个五十多岁的中年妇女，声音略带沙哑，"
            "嗓门大，热情泼辣，像一个心直口快的阿姨。"
            "<|endofprompt|>"
        ),
    },
    "顾明轩": {
        "gender": "male",
        "desc": "渣男前夫，纨绔少爷",
        "sample_text": "若溪，你听我解释，甜甜她不是你想的那种人！你怎么就不能理解我呢？",
        "mode": "instruct2",
        "ref_wav": "ref_male_arrogant.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个二十多岁的年轻男性，声音偏高、尖锐，"
            "语气傲慢急躁，带有不屑和轻蔑，像一个纨绔子弟。"
            "<|endofprompt|>"
        ),
    },
    "陈甜甜": {
        "gender": "female",
        "desc": "小三/绿茶反派，心机女",
        "sample_text": "明轩哥哥，我真的怀孕了，是个男孩呢！你不要离开我好不好？",
        "mode": "instruct2",
        "ref_wav": "ref_female_sweet_fake.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个二十多岁的年轻女性，声音甜美嗲气，"
            "刻意装可怜，带着做作和心机的感觉。"
            "<|endofprompt|>"
        ),
    },
    "李氏": {
        "gender": "female",
        "desc": "婆婆，尖酸刻薄老太太",
        "sample_text": "嫁进我们顾家，就得守我们顾家的规矩！生不出儿子，还有什么脸待在这个家里！",
        "mode": "instruct2",
        "ref_wav": "ref_female_sharp.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个六十多岁的老年女性，声音尖锐、刺耳、刻薄，"
            "语速快，充满嫌弃和刻薄，像一个尖酸的老太太。"
            "<|endofprompt|>"
        ),
    },
    "林书远": {
        "gender": "male",
        "desc": "女主父亲，林氏集团董事长",
        "sample_text": "若溪，当年是爸爸对不起你。但你要相信，我做的一切都是为了这个家。",
        "mode": "instruct2",
        "ref_wav": "ref_male_authoritative.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个五十多岁的中年男性企业家，声音浑厚威严，"
            "充满权威感，语速缓慢沉稳，像一个久经商场的集团董事长。"
            "<|endofprompt|>"
        ),
    },
    "陈思琪": {
        "gender": "female",
        "desc": "海外投资经理，干练女精英",
        "sample_text": "林总，柬埔寨那边有个移动支付项目，我觉得非常有潜力，值得重点跟进。",
        "mode": "instruct2",
        "ref_wav": "ref_female_pro.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个三十岁左右的职业女性，声音干练、清晰、自信，"
            "语速较快，条理清晰，像一个精英投资经理。"
            "<|endofprompt|>"
        ),
    },
    "叙述者": {
        "gender": "neutral",
        "desc": "旁白/叙述者，中性沉稳",
        "sample_text": "金七七在天庭待了三百年。三百年来，她每天的工作就是坐在财运司的办公桌后面，翻看人间各地的财运报表。",
        "mode": "instruct2",
        "ref_wav": "ref_narrator.wav",
        "instruct_text": (
            "You are a helpful assistant. "
            "说话者是一个三十多岁的女性播音员，声音中性沉稳、清晰流畅，"
            "语速均匀，带有讲述感，像一个专业的有声书演播者。"
            "<|endofprompt|>"
        ),
    },
}

# CosyVoice3 自带的默认参考音频（在仓库 asset 目录下）
DEFAULT_PROMPT_WAV = os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

# ============================================================
# CosyVoice3 引擎（延迟加载）
# ============================================================
_model = None


def get_engine():
    global _model
    if _model is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        print(f"Loading CosyVoice3 model from {MODEL_DIR}...")
        _model = AutoModel(model_dir=MODEL_DIR)
        spks = _model.list_available_spks()
        print(f"Model loaded. Available speakers: {spks}")
    return _model


def _get_prompt_wav(char_info):
    """获取角色的参考音频路径"""
    # 优先使用角色专属 ref_wav（生成的参考音频）
    ref_file = char_info.get("ref_wav", "")
    if ref_file:
        ref_path = os.path.join(PROMPT_DIR, ref_file)
        if os.path.exists(ref_path):
            return ref_path
    # 其次使用用户提供的 prompt_wav（zero_shot 模式）
    prompt_file = char_info.get("prompt_wav", "")
    if prompt_file:
        local_path = os.path.join(PROMPT_DIR, prompt_file)
        if os.path.exists(local_path):
            return local_path
    # 回退到默认参考音频
    if os.path.exists(DEFAULT_PROMPT_WAV):
        return DEFAULT_PROMPT_WAV
    return None


# ============================================================
# 文本解析：把章节内容拆成角色对话段
# ============================================================
def parse_chapter_segments(md_text):
    """
    将章节markdown拆分为多个片段，每个片段标记角色。
    返回: [(role, text), ...]
    """
    lines = md_text.strip().split('\n')
    segments = []

    name_map = {
        "小七": "小七", "七七": "小七", "金七七": "小七", "念七": "小七",
        "林若溪": "林若溪", "若溪": "林若溪",
        "陆北辰": "陆北辰", "北辰": "陆北辰",
        "王妈": "王妈",
        "顾明轩": "顾明轩", "明轩": "顾明轩",
        "陈甜甜": "陈甜甜", "甜甜": "陈甜甜",
        "李氏": "李氏", "婆婆": "李氏",
        "林书远": "林书远", "书远": "林书远",
        "陈思琪": "陈思琪", "思琪": "陈思琪",
    }

    current_role = "叙述者"
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    segments.append((current_role, text))
                buffer = []
            current_role = "叙述者"
            continue

        if stripped == '---':
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    segments.append((current_role, text))
                buffer = []
            current_role = "叙述者"
            continue

        # 【...】内心OS块
        if stripped.startswith('【'):
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    segments.append((current_role, text))
                buffer = []
            os_text = stripped.strip('【】')
            if '妈' in os_text or '妈妈' in os_text or '妈咪' in os_text:
                segments.append(("小七", os_text))
            elif '小七' in os_text or '七七' in os_text or '念七' in os_text:
                segments.append(("林若溪", os_text))
            else:
                segments.append(("小七", os_text))
            current_role = "叙述者"
            continue

        # 检测对话行
        said_match = re.match(
            r'(.{2,6})[说道笑道冷笑哼着叫喊问回答低声]?\s*[：:]\s*[""\u201c\u300c]?(.+?)[""\u201d\u300d]?\s*$',
            stripped
        )
        if not said_match:
            said_match2 = re.match(
                r'[""\u201c\u300c](.+?)[""\u201d\u300d]\s*[—]?\s*(.{2,6})[说道笑道冷笑低声]?\s*$',
                stripped
            )
            if said_match2:
                dialog_text = said_match2.group(1)
                speaker_name = said_match2.group(2)
                if buffer:
                    text = '\n'.join(buffer).strip()
                    if text:
                        segments.append((current_role, text))
                    buffer = []
                role = name_map.get(speaker_name, "叙述者")
                segments.append((role, dialog_text))
                current_role = "叙述者"
                continue

        if said_match:
            speaker_name = said_match.group(1)
            dialog_text = said_match.group(2)
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    segments.append((current_role, text))
                buffer = []
            role = name_map.get(speaker_name, "叙述者")
            segments.append((role, dialog_text))
            current_role = "叙述者"
            continue

        # 纯引号对话
        quote_match = re.match(r'^[""\u201c\u300c](.+?)[""\u201d\u300d]$', stripped)
        if quote_match:
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    segments.append((current_role, text))
                buffer = []
            segments.append((current_role, quote_match.group(1)))
            continue

        # 普通叙述文本
        buffer.append(stripped)
        current_role = "叙述者"

    # flush
    if buffer:
        text = '\n'.join(buffer).strip()
        if text:
            segments.append((current_role, text))

    # 合并相邻同角色片段
    merged = []
    for role, text in segments:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + '\n' + text)
        else:
            merged.append((role, text))

    return merged


# ============================================================
# 语音合成 - CosyVoice3 版
# ============================================================
def clean_text(text):
    """清理文本，保留 CosyVoice 支持的字符"""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            result.append(ch)
        elif ch in '，。、；：！？：\n':
            result.append(ch)
        elif ch.isascii() and (ch.isalnum() or ch.isspace()):
            result.append(ch)
        elif ch in '.,!?;:\'-"()~@#$%^&*+=<>[]{}/\\':
            result.append(ch)
        elif ch in '—－–':
            result.append(',')
        elif ch in '………':
            result.append('。')
        elif ch in '\u201c\u201d':
            result.append('"')
        elif ch in '\u2018\u2019':
            result.append("'")
        elif ch in '·\t':
            result.append(ch)
    return ''.join(result)


def _split_text(text, max_len=200):
    """长文本按标点分段"""
    if len(text) <= max_len:
        return [text]
    chunks = []
    sentences = re.split(r'([。！？；\n])', text)
    current = ""
    for s in sentences:
        current += s
        if len(current) >= max_len:
            if current.strip():
                chunks.append(current)
            current = ""
    if current.strip():
        chunks.append(current)
    return chunks


def _synthesize_chunk_zero_shot(model, chunk, prompt_text, prompt_wav):
    """zero_shot 模式合成单段"""
    # CosyVoice3 要求 prompt_text 包含 <|endofprompt|> 标记
    if '<|endofprompt|>' not in prompt_text:
        prompt_text = prompt_text.rstrip() + ' <|endofprompt|>'
    all_audio = []
    for j in model.inference_zero_shot(
        chunk, prompt_text, prompt_wav, stream=False
    ):
        speech = j['tts_speech'].cpu()
        if speech.dim() > 1:
            speech = speech.squeeze(0)
        all_audio.append(speech)
    return all_audio


def _synthesize_chunk_instruct2(model, chunk, instruct_text, prompt_wav):
    """instruct2 模式合成单段"""
    all_audio = []
    for j in model.inference_instruct2(
        chunk, instruct_text, prompt_wav, stream=False
    ):
        speech = j['tts_speech'].cpu()
        if speech.dim() > 1:
            speech = speech.squeeze(0)
        all_audio.append(speech)
    return all_audio


def synthesize(text, character, output_path):
    """用 CosyVoice3 合成单段语音"""
    import torchaudio
    import torch

    model = get_engine()
    char_info = CHARACTERS.get(character, CHARACTERS["叙述者"])
    mode = char_info.get("mode", "instruct2")

    safe_text = clean_text(text)
    if not safe_text.strip():
        return None

    chunks = _split_text(safe_text)
    all_audio = []

    # 准备推理参数
    if mode == "zero_shot":
        prompt_text = char_info.get("prompt_text", DEFAULT_PROMPT_TEXT)
        prompt_wav = _get_prompt_wav(char_info)
        if not prompt_wav:
            # 回退到 instruct2
            mode = "instruct2"

    if mode == "instruct2":
        instruct_text = char_info.get(
            "instruct_text",
            "You are a helpful assistant.<|endofprompt|>"
        )
        prompt_wav = _get_prompt_wav(char_info)
        if not prompt_wav and os.path.exists(DEFAULT_PROMPT_WAV):
            prompt_wav = DEFAULT_PROMPT_WAV

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            print(f"    Synthesizing ({mode}): {chunk[:40]}...")
            if mode == "zero_shot":
                audio_parts = _synthesize_chunk_zero_shot(
                    model, chunk, prompt_text, prompt_wav
                )
            else:
                audio_parts = _synthesize_chunk_instruct2(
                    model, chunk, instruct_text, prompt_wav
                )
            all_audio.extend(audio_parts)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    OOM, retrying with shorter segments...")
                torch.cuda.empty_cache()
                # 切短重试
                sub_parts = re.split(r'([，。、；：！？])', chunk)
                sub_current = ""
                for sp in sub_parts:
                    sub_current += sp
                    if len(sub_current) >= 60:
                        try:
                            if mode == "zero_shot":
                                parts = _synthesize_chunk_zero_shot(
                                    model, sub_current.strip(), prompt_text, prompt_wav
                                )
                            else:
                                parts = _synthesize_chunk_instruct2(
                                    model, sub_current.strip(), instruct_text, prompt_wav
                                )
                            all_audio.extend(parts)
                        except Exception as e2:
                            print(f"    Sub-chunk failed: {e2}")
                        sub_current = ""
                if sub_current.strip():
                    try:
                        if mode == "zero_shot":
                            parts = _synthesize_chunk_zero_shot(
                                model, sub_current.strip(), prompt_text, prompt_wav
                            )
                        else:
                            parts = _synthesize_chunk_instruct2(
                                model, sub_current.strip(), instruct_text, prompt_wav
                            )
                        all_audio.extend(parts)
                    except Exception:
                        pass
            else:
                raise

    if not all_audio:
        return None

    audio_tensor = torch.cat(all_audio, dim=0)
    torchaudio.save(output_path, audio_tensor.unsqueeze(0), model.sample_rate)
    duration = audio_tensor.shape[0] / model.sample_rate
    print(f"    Saved: {output_path} ({duration:.1f}s)")
    return output_path


def synthesize_chapter(chapter_file, force=False):
    """合成整章语音"""
    basename = os.path.basename(chapter_file).replace('.md', '')
    chapter_audio_dir = os.path.join(AUDIO_DIR, basename)
    os.makedirs(chapter_audio_dir, exist_ok=True)

    manifest_path = os.path.join(chapter_audio_dir, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        return json.load(open(manifest_path, 'r', encoding='utf-8'))

    with open(chapter_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    segments = parse_chapter_segments(md_text)
    manifest = {"chapter": basename, "segments": []}

    for i, (role, text) in enumerate(segments):
        seg_file = f"seg_{i:03d}_{role}.wav"
        seg_path = os.path.join(chapter_audio_dir, seg_file)

        print(f"  [{i+1}/{len(segments)}] {role}: {text[:30]}...")
        try:
            synthesize(text, role, seg_path)
            manifest["segments"].append({
                "index": i,
                "role": role,
                "text": text[:100],
                "file": seg_file,
            })
        except Exception as e:
            print(f"  Failed: {e}")
            manifest["segments"].append({
                "index": i,
                "role": role,
                "text": text[:100],
                "file": None,
                "error": str(e),
            })

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def synthesize_samples():
    """为每个角色生成试听样本"""
    sample_dir = os.path.join(AUDIO_DIR, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    manifest = {}
    for name, info in CHARACTERS.items():
        output = os.path.join(sample_dir, f"{name}.wav")
        mode = info.get("mode", "instruct2")
        print(f"Generating sample: {name} ({mode})...")
        try:
            synthesize(info["sample_text"], name, output)
            manifest[name] = {
                "gender": info["gender"],
                "mode": mode,
                "desc": info["desc"],
                "sample_text": info["sample_text"],
                "file": f"samples/{name}.wav",
            }
            print(f"  Done: {name}")
        except Exception as e:
            print(f"  Failed: {name} - {e}")
            manifest[name] = {"error": str(e)}

    manifest_path = os.path.join(AUDIO_DIR, "samples", "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSamples generated: {sample_dir}")
    return manifest


# ============================================================
# CLI 入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Novel TTS - CosyVoice3')
    parser.add_argument('--samples', action='store_true', help='Generate character samples')
    parser.add_argument('--chapter', type=str, help='Synthesize a chapter')
    parser.add_argument('--all', action='store_true', help='Synthesize all chapters')
    parser.add_argument('--serve', action='store_true', help='Start TTS API server')
    parser.add_argument('--port', type=int, default=8084, help='TTS server port')
    args = parser.parse_args()

    if args.samples:
        synthesize_samples()
    elif args.chapter:
        ch_arg = args.chapter
        if not os.path.exists(ch_arg):
            candidates = [
                f for f in os.listdir(os.path.join(NOVEL_DIR, "chapters"))
                if f.startswith(ch_arg) and f.endswith('.md')
            ]
            if candidates:
                ch_arg = os.path.join(NOVEL_DIR, "chapters", candidates[0])
            else:
                print(f"Chapter not found: {args.chapter}")
                sys.exit(1)
        synthesize_chapter(ch_arg)
    elif args.all:
        chapters_dir = os.path.join(NOVEL_DIR, "chapters")
        for f in sorted(os.listdir(chapters_dir)):
            if f.endswith('.md'):
                print(f"\nSynthesizing: {f}")
                synthesize_chapter(os.path.join(chapters_dir, f))
    elif args.serve:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse

        class TTSHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)

                if parsed.path == '/api/characters':
                    data = json.dumps(CHARACTERS, ensure_ascii=False)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data.encode('utf-8'))

                elif parsed.path == '/api/synthesize_sample':
                    name = params.get('name', ['叙述者'])[0]
                    char_info = CHARACTERS.get(name, CHARACTERS['叙述者'])
                    text = params.get('text', [char_info['sample_text']])[0]
                    print(f"Sample request: {name} ({char_info.get('mode', 'instruct2')}) - {text[:30]}...")

                    import tempfile
                    temp_dir = os.path.join(AUDIO_DIR, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        suffix='.wav', delete=False, dir=temp_dir
                    ) as tmp:
                        tmp_path = tmp.name

                    try:
                        synthesize(text, name, tmp_path)
                        with open(tmp_path, 'rb') as f:
                            data = f.read()
                        print(f"  Sample audio: {len(data)} bytes")
                        self.send_response(200)
                        self.send_header('Content-Type', 'audio/wav')
                        self.send_header('Content-Length', len(data))
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(data)
                    except Exception as e:
                        print(f"  Sample failed: {e}")
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': str(e)}).encode())
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == '/api/synthesize':
                    length = int(self.headers.get('Content-Length', 0))
                    body = json.loads(self.rfile.read(length).decode('utf-8'))
                    text = body.get('text', '')
                    character = body.get('character', '叙述者')

                    import tempfile
                    temp_dir = os.path.join(AUDIO_DIR, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        suffix='.wav', delete=False, dir=temp_dir
                    ) as tmp:
                        tmp_path = tmp.name

                    try:
                        synthesize(text, character, tmp_path)
                        with open(tmp_path, 'rb') as f:
                            data = f.read()
                        self.send_response(200)
                        self.send_header('Content-Type', 'audio/wav')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(data)
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': str(e)}).encode())
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

                elif self.path == '/api/synthesize-samples':
                    manifest = synthesize_samples()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(manifest, ensure_ascii=False).encode())

                elif self.path.startswith('/api/synthesize-chapter/'):
                    chapter_name = self.path.split('/')[-1]
                    chapters_dir = os.path.join(NOVEL_DIR, "chapters")
                    chapter_file = None
                    for f in os.listdir(chapters_dir):
                        if f.startswith(chapter_name) and f.endswith('.md'):
                            chapter_file = os.path.join(chapters_dir, f)
                            break
                    if chapter_file:
                        manifest = synthesize_chapter(chapter_file)
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps(manifest, ensure_ascii=False).encode())
                    else:
                        self.send_error(404, 'Chapter not found')

                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                print(f"[TTS] {format % args}")

        print(f"CosyVoice3 TTS API server: http://0.0.0.0:{args.port}", flush=True)
        server = HTTPServer(('0.0.0.0', args.port), TTSHandler)
        server.serve_forever()
    else:
        parser.print_help()
