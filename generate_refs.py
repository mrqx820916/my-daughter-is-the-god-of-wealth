#!/usr/bin/env python3
"""
为各角色生成专属参考音频
用 CosyVoice3 instruct2 模式 + 强化指令生成不同音色的 base reference，
后续合成时用这些 reference 作为 prompt_wav，音色差异会大很多。
"""
import os, sys

COSYVOICE_DIR = os.environ.get("COSYVOICE_DIR", r"D:\CosyVoice")
sys.path.insert(0, os.path.join(COSYVOICE_DIR, 'third_party/Matcha-TTS'))
sys.path.insert(0, COSYVOICE_DIR)

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
DEFAULT_WAV = os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 各声线的参考文本和强化指令
# 关键：用非常具体的中文描述音色特征，让模型区分
VOICE_PROFILES = {
    "ref_child_girl": {
        "text": "妈咪，你看那个气球好漂亮呀，我想要那个粉色的！",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个四五岁的小女孩，声音非常稚嫩、清脆、甜美、可爱，"
            "带着奶声奶气的童真感，语调活泼上扬，充满天真和快乐。"
            "<|endofprompt|>"
        ),
    },
    "ref_female_gentle": {
        "text": "没事的，一切都会好起来的。我会一直在你身边，不管发生什么。",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个二十五六岁的年轻女性，声音温柔、舒缓、柔和，"
            "带有坚定的力量感，语速适中，充满母性的温暖和关怀。"
            "<|endofprompt|>"
        ),
    },
    "ref_male_deep": {
        "text": "这件事我自有安排，你不必担心。该来的总会来。",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个三十多岁的成熟男性，声音低沉、浑厚、富有磁性，"
            "音调偏低，语速沉稳，充满自信和掌控力，像商业精英。"
            "<|endofprompt|>"
        ),
    },
    "ref_male_arrogant": {
        "text": "你算什么东西？也配跟我说这种话？真是笑话！",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个二十多岁的年轻男性，声音偏高、尖锐，"
            "语气傲慢急躁，带有不屑和轻蔑，像一个纨绔子弟。"
            "<|endofprompt|>"
        ),
    },
    "ref_male_authoritative": {
        "text": "这个决定我已经做完了，不需要再讨论。按我说的去做。",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个五十多岁的中年男性企业家，声音浑厚威严，"
            "充满权威感，语速缓慢沉稳，像一个久经商场的集团董事长。"
            "<|endofprompt|>"
        ),
    },
    "ref_female_sharp": {
        "text": "哼，就凭她那种出身，也配进我们家的门？真是笑死人了！",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个六十多岁的老年女性，声音尖锐、刺耳、刻薄，"
            "语速快，充满嫌弃和刻薄，像一个尖酸的老太太。"
            "<|endofprompt|>"
        ),
    },
    "ref_auntie": {
        "text": "小姐您放心，有我老婆子在，谁也别想欺负您！我这就去收拾她！",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个五十多岁的中年妇女，声音略带沙哑，"
            "嗓门大，热情泼辣，像一个心直口快的阿姨。"
            "<|endofprompt|>"
        ),
    },
    "ref_female_sweet_fake": {
        "text": "哥哥，人家不是故意的嘛，你就别生气了好不好？嗯~",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个二十多岁的年轻女性，声音甜美嗲气，"
            "刻意装可怜，带着做作和心机的感觉。"
            "<|endofprompt|>"
        ),
    },
    "ref_female_pro": {
        "text": "根据我的分析，这个项目的投资回报率在百分之二十以上，建议立即推进。",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个三十岁左右的职业女性，声音干练、清晰、自信，"
            "语速较快，条理清晰，像一个精英投资经理。"
            "<|endofprompt|>"
        ),
    },
    "ref_narrator": {
        "text": "夕阳的余晖洒在城市的天际线上，一切看似平静，暗涌却已在悄然酝酿。",
        "instruct": (
            "You are a helpful assistant. "
            "说话者是一个三十多岁的女性播音员，声音中性沉稳、清晰流畅，"
            "语速均匀，带有讲述感，像一个专业的有声书演播者。"
            "<|endofprompt|>"
        ),
    },
}

# 角色 → 参考音频映射
CHAR_REF_MAP = {
    "小七": "ref_child_girl.wav",
    "林若溪": "ref_female_gentle.wav",
    "陆北辰": "ref_male_deep.wav",
    "王妈": "ref_auntie.wav",
    "顾明轩": "ref_male_arrogant.wav",
    "陈甜甜": "ref_female_sweet_fake.wav",
    "李氏": "ref_female_sharp.wav",
    "林书远": "ref_male_authoritative.wav",
    "陈思琪": "ref_female_pro.wav",
    "叙述者": "ref_narrator.wav",
}


def generate_refs():
    print(f"Loading model from {MODEL_DIR}...")
    model = AutoModel(model_dir=MODEL_DIR)
    print(f"Model loaded. Sample rate: {model.sample_rate}")

    for name, profile in VOICE_PROFILES.items():
        output_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        if os.path.exists(output_path):
            print(f"  [SKIP] {name} (already exists)")
            continue

        print(f"  Generating {name}...")
        all_audio = []
        for j in model.inference_instruct2(
            profile["text"],
            profile["instruct"],
            DEFAULT_WAV,
            stream=False
        ):
            speech = j['tts_speech'].cpu()
            if speech.dim() > 1:
                speech = speech.squeeze(0)
            all_audio.append(speech)

        if all_audio:
            audio_tensor = torch.cat(all_audio, dim=0)
            torchaudio.save(output_path, audio_tensor.unsqueeze(0), model.sample_rate)
            duration = audio_tensor.shape[0] / model.sample_rate
            print(f"    -> {output_path} ({duration:.1f}s)")
        else:
            print(f"    -> FAILED: no audio generated")

    # 打印角色映射
    print(f"\n角色→参考音频映射:")
    for char, ref in CHAR_REF_MAP.items():
        print(f"  {char}: {ref}")

    print(f"\n参考音频目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_refs()
