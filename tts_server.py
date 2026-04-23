#!/usr/bin/env python3
"""
小说TTS语音合成服务器 - CosyVoice多角色语音
用法: source ~/xiaoshuo-venv/bin/activate && python tts_server.py --serve --port 8084
依赖: CosyVoice (git clone), torch, torchaudio, modelscope等
"""
import os, sys, json, re, time, argparse

# CosyVoice需要添加路径
COSYVOICE_DIR = "/mnt/zspace/worksp/CosyVoice"
sys.path.insert(0, os.path.join(COSYVOICE_DIR, 'third_party/Matcha-TTS'))
sys.path.insert(0, COSYVOICE_DIR)

import numpy as np

NOVEL_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(NOVEL_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

MODEL_DIR = "/root/cosyvoice_models/CosyVoice-300M-SFT"

# ============================================================
# 角色声音配置 - CosyVoice SFT模式
# 中文女/中文男 是预设角色，男女声分离
# ============================================================
# SFT角色映射：每个小说角色 → CosyVoice SFT speaker
SFT_VOICE_MAP = {
    "female": "中文女",
    "male":   "中文男",
}

CHARACTERS = {
    "小七": {
        "gender": "female",
        "desc": "天庭财神/女儿，童声但老灵魂",
        "sample_text": "妈咪，那个垃圾股其实是个宝贝哦~本财神当年点拨过范蠡，教过沈万三呢！",
        "voice": "中文女",
    },
    "林若溪": {
        "gender": "female",
        "desc": "女主/妈妈，温柔坚定",
        "sample_text": "我什么都不要，只要我的孩子。从今天起，我不会再让任何人欺负我们母女。",
        "voice": "中文女",
    },
    "陆北辰": {
        "gender": "male",
        "desc": "男主，腹黑深情商界精英",
        "sample_text": "你是我见过唯一一个让数据说谎的人。林若溪，你让我很感兴趣。",
        "voice": "中文男",
    },
    "王妈": {
        "gender": "female",
        "desc": "忠心保姆，泼辣心善，偏老年女声",
        "sample_text": "小姐，您放心，有我王妈在，谁也别想欺负您和小姐！",
        "voice": "中文女",
    },
    "顾明轩": {
        "gender": "male",
        "desc": "渣男前夫，纨绔少爷",
        "sample_text": "若溪，你听我解释，甜甜她不是你想的那种人！你怎么就不能理解我呢？",
        "voice": "中文男",
    },
    "陈甜甜": {
        "gender": "female",
        "desc": "小三/绿茶反派，心机女",
        "sample_text": "明轩哥哥，我真的怀孕了，是个男孩呢！你不要离开我好不好？",
        "voice": "中文女",
    },
    "李氏": {
        "gender": "female",
        "desc": "婆婆，尖酸刻薄老太太",
        "sample_text": "嫁进我们顾家，就得守我们顾家的规矩！生不出儿子，还有什么脸待在这个家里！",
        "voice": "中文女",
    },
    "林书远": {
        "gender": "male",
        "desc": "女主父亲，林氏集团董事长",
        "sample_text": "若溪，当年是爸爸对不起你。但你要相信，我做的一切都是为了这个家。",
        "voice": "中文男",
    },
    "陈思琪": {
        "gender": "female",
        "desc": "海外投资经理，干练女精英",
        "sample_text": "林总，柬埔寨那边有个移动支付项目，我觉得非常有潜力，值得重点跟进。",
        "voice": "中文女",
    },
    "叙述者": {
        "gender": "neutral",
        "desc": "旁白/叙述者，中性沉稳",
        "sample_text": "金七七在天庭待了三百年。三百年来，她每天的工作就是坐在财运司的办公桌后面，翻看人间各地的财运报表。",
        "voice": "中文女",  # 叙述者用女声偏柔和
    },
}

# ============================================================
# CosyVoice 引擎（延迟加载）
# ============================================================
_model = None

def get_engine():
    global _model
    if _model is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        print("⏳ 正在加载CosyVoice-300M-SFT模型...")
        _model = AutoModel(model_dir=MODEL_DIR)
        print(f"✅ CosyVoice模型加载完成，可用角色: {_model.list_available_spks()}")
    return _model

# ============================================================
# 文本解析：把章节内容拆成角色对话段
# ============================================================
def parse_chapter_segments(md_text):
    """
    将章节markdown拆分为多个片段，每个片段标记角色。
    返回: [(role, text), ...]
    
    规则:
    1. 「角色名」开头的引号对话 → 对应角色
    2. 【...】内心OS → 小七或林若溪（根据上下文判断）
    3. 其他 → 叙述者
    """
    lines = md_text.strip().split('\n')
    segments = []
    
    # 角色名映射：处理各种称呼
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
            # 空行或标题，flush buffer
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
            # 内心OS归属小七（默认）或林若溪
            os_text = stripped.strip('【】')
            if '妈' in os_text or '妈妈' in os_text or '妈咪' in os_text:
                segments.append(("小七", os_text))
            elif '小七' in os_text or '七七' in os_text or '念七' in os_text:
                segments.append(("林若溪", os_text))
            else:
                segments.append(("小七", os_text))  # 大部分内心OS是小七
            current_role = "叙述者"
            continue
        
        # 检测对话行: "xxx" 或 "xxx"说
        # 格式: 林若溪说："xxx" / "xxx" / "xxx。"林若溪说
        said_match = re.match(r'(.{2,6})[说道笑道冷笑哼着叫喊问回答低声]?\s*[：:]\s*[""「]?(.+?)[""「]?\s*$', stripped)
        if not said_match:
            # 反向格式: "xxx"角色说
            said_match2 = re.match(r'[""「](.+?)[""」]\s*[—]?\s*(.{2,6})[说道笑道冷笑低声]?\s*$', stripped)
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
        
        # 纯引号对话（无角色名标注）
        quote_match = re.match(r'^[""「](.+?)[""」]$', stripped)
        if quote_match:
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    segments.append((current_role, text))
                buffer = []
            segments.append((current_role if current_role != "叙述者" else "叙述者", quote_match.group(1)))
            continue
        
        # 普通叙述文本
        buffer.append(stripped)
        current_role = "叙述者"
    
    # flush
    if buffer:
        text = '\n'.join(buffer).strip()
        if text:
            segments.append((current_role, text))
    
    # 合并相邻的同角色片段
    merged = []
    for role, text in segments:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + '\n' + text)
        else:
            merged.append((role, text))
    
    return merged

# ============================================================
# 语音合成 - CosyVoice版
# ============================================================
def clean_text(text):
    """清理文本，移除CosyVoice不支持的字符"""
    result = []
    for ch in text:
        cp = ord(ch)
        # 中文汉字 + 中文标点
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            result.append(ch)
        elif ch in '，。、；：！？：\n':
            result.append(ch)
        # 英文字母数字空格
        elif ch.isascii() and (ch.isalnum() or ch.isspace()):
            result.append(ch)
        # 基础标点
        elif ch in '.,!?;:\'-"()~@#$%^&*+=<>[]{}/\\':
            result.append(ch)
        # 特殊替换
        elif ch in '—－–':
            result.append(',')
        elif ch in '………':
            result.append('。')
        elif ch in '\u201c\u201d':
            result.append('"')
        elif ch in '\u2018\u2019':
            result.append("'")
        elif ch in '【】《》〈〉（）~':
            pass  # 丢弃
        elif ch in '·\t':
            result.append(ch)
        # 其他跳过（emoji等）
    return ''.join(result)


def synthesize(text, character, output_path):
    """用CosyVoice合成单段语音"""
    import torchaudio
    
    model = get_engine()
    char_info = CHARACTERS.get(character, CHARACTERS["叙述者"])
    sft_spk = char_info["voice"]
    
    # 清理文本
    safe_text = clean_text(text)
    if not safe_text.strip():
        return None
    
    # CosyVoice SFT对长文本分段（每段不超过200字）
    max_len = 200
    chunks = []
    if len(safe_text) <= max_len:
        chunks = [safe_text]
    else:
        # 按标点分段
        sentences = re.split(r'([。！？；\n])', safe_text)
        current = ""
        for s in sentences:
            current += s
            if len(current) >= max_len:
                if current.strip():
                    chunks.append(current)
                current = ""
        if current.strip():
            chunks.append(current)
    
    all_audio = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            print(f"    CosyVoice合成: {chunk[:40]}...")
            for i, j in enumerate(model.inference_sft(chunk, sft_spk, stream=False)):
                # j['tts_speech'] 是 torch.Tensor, shape (1, samples)
                speech = j['tts_speech'].cpu()
                if speech.dim() > 1:
                    speech = speech.squeeze(0)
                all_audio.append(speech)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    ⚠️ OOM，切短重试: {chunk[:30]}...")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # 切成更小的段
                sub_parts = re.split(r'([，。、；：！？])', chunk)
                sub_current = ""
                for sp in sub_parts:
                    sub_current += sp
                    if len(sub_current) >= 60:
                        try:
                            for i, j in enumerate(model.inference_sft(sub_current.strip(), sft_spk, stream=False)):
                                speech = j['tts_speech'].cpu()
                                if speech.dim() > 1:
                                    speech = speech.squeeze(0)
                                all_audio.append(speech)
                        except Exception as e2:
                            print(f"    ⚠️ 子段也失败: {e2}")
                        sub_current = ""
                if sub_current.strip():
                    try:
                        for i, j in enumerate(model.inference_sft(sub_current.strip(), sft_spk, stream=False)):
                            speech = j['tts_speech'].cpu()
                            if speech.dim() > 1:
                                speech = speech.squeeze(0)
                            all_audio.append(speech)
                    except Exception:
                        pass
            else:
                raise
    
    if not all_audio:
        return None
    
    # 拼接音频
    import torch
    audio_tensor = torch.cat(all_audio, dim=0)
    
    # 保存为wav
    torchaudio.save(output_path, audio_tensor.unsqueeze(0), model.sample_rate)
    print(f"    ✅ 已保存: {output_path} ({audio_tensor.shape[0]/model.sample_rate:.1f}s)")
    return output_path


def synthesize_chapter(chapter_file, force=False):
    """合成整章语音"""
    basename = os.path.basename(chapter_file).replace('.md', '')
    chapter_audio_dir = os.path.join(AUDIO_DIR, basename)
    os.makedirs(chapter_audio_dir, exist_ok=True)
    
    # 检查是否已合成
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
        
        print(f"  合成 [{i+1}/{len(segments)}] {role}: {text[:30]}...")
        try:
            synthesize(text, role, seg_path)
            manifest["segments"].append({
                "index": i,
                "role": role,
                "text": text[:100],
                "file": seg_file,
            })
        except Exception as e:
            print(f"  ⚠️ 合成失败: {e}")
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
        print(f"🎙️ 合成 {name} ({info['voice']}) 试听...")
        try:
            synthesize(info["sample_text"], name, output)
            manifest[name] = {
                "gender": info["gender"],
                "voice": info["voice"],
                "desc": info["desc"],
                "sample_text": info["sample_text"],
                "file": f"samples/{name}.wav",
            }
            print(f"  ✅ {name} 完成")
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            manifest[name] = {"error": str(e)}
    
    manifest_path = os.path.join(AUDIO_DIR, "samples", "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 试听样本生成完成: {sample_dir}")
    return manifest


# ============================================================
# CLI 入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='小说TTS语音合成 - CosyVoice版')
    parser.add_argument('--samples', action='store_true', help='生成角色试听样本')
    parser.add_argument('--chapter', type=str, help='合成指定章节')
    parser.add_argument('--all', action='store_true', help='合成全部章节')
    parser.add_argument('--serve', action='store_true', help='启动TTS API服务器')
    parser.add_argument('--port', type=int, default=8084, help='TTS服务端口')
    args = parser.parse_args()
    
    if args.samples:
        synthesize_samples()
    elif args.chapter:
        # 支持章节ID（如V01C001）或完整路径
        ch_arg = args.chapter
        if not os.path.exists(ch_arg):
            # 尝试在chapters目录中查找
            candidates = [f for f in os.listdir(os.path.join(NOVEL_DIR, "chapters"))
                          if f.startswith(ch_arg) and f.endswith('.md')]
            if candidates:
                ch_arg = os.path.join(NOVEL_DIR, "chapters", candidates[0])
            else:
                print(f"❌ 找不到章节: {args.chapter}")
                sys.exit(1)
        synthesize_chapter(ch_arg)
    elif args.all:
        chapters_dir = os.path.join(NOVEL_DIR, "chapters")
        for f in sorted(os.listdir(chapters_dir)):
            if f.endswith('.md'):
                print(f"\n📖 合成: {f}")
                synthesize_chapter(os.path.join(chapters_dir, f))
    elif args.serve:
        # TTS API 服务器模式
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        class TTSHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                
                if parsed.path == '/api/characters':
                    # 返回角色列表
                    data = json.dumps(CHARACTERS, ensure_ascii=False)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data.encode('utf-8'))
                
                elif parsed.path == '/api/synthesize_sample':
                    # 试听API
                    name = params.get('name', ['叙述者'])[0]
                    char_info = CHARACTERS.get(name, CHARACTERS['叙述者'])
                    text = params.get('text', [char_info['sample_text']])[0]
                    print(f"🎧 试听请求: {name} ({char_info['voice']}) - {text[:30]}...")
                    
                    import tempfile
                    temp_dir = os.path.join(AUDIO_DIR, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir) as tmp:
                        tmp_path = tmp.name
                    
                    try:
                        synthesize(text, name, tmp_path)
                        with open(tmp_path, 'rb') as f:
                            data = f.read()
                        print(f"  ✅ 试听音频: {len(data)} bytes")
                        self.send_response(200)
                        self.send_header('Content-Type', 'audio/wav')
                        self.send_header('Content-Length', len(data))
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(data)
                    except Exception as e:
                        print(f"  ❌ 试听失败: {e}")
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
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir) as tmp:
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
        
        print(f"🎙️ CosyVoice TTS API服务器启动: http://0.0.0.0:{args.port}", flush=True)
        server = HTTPServer(('0.0.0.0', args.port), TTSHandler)
        server.serve_forever()
    else:
        parser.print_help()
