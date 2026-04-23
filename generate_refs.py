#!/usr/bin/env python3
"""
为各角色生成专属参考音频
用微软 Edge TTS 生成真正不同声线的参考音频（男声/女声/童声/老年声），
再用 librosa 重采样到 CosyVoice3 要求的 24000Hz。
"""
import os, sys, asyncio

import soundfile as sf
import numpy as np
import librosa

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "prompts")
TARGET_SR = 24000  # CosyVoice3 采样率
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Edge TTS 中文语音列表：
#   男声: YunxiNeural(年轻), YunjianNeural(成熟), YunyangNeural(新闻主播)
#   女声: XiaoxiaoNeural(温柔), XiaoyiNeural(活泼), XiaomoNeural(成熟)
#         XiaoruiNeural(老年), XiaoshuangNeural(儿童), XiaozhenNeural(知性)
#         XiaochenNeural(干练)

# 角色 → (Edge TTS voice, 参考文本)
VOICE_PROFILES = {
    # 小七：儿童女声 → 活泼女声 XiaoyiNeural（最接近童声）
    "ref_child_girl": (
        "zh-CN-XiaoyiNeural",
        "妈咪你看那个气球好漂亮呀，我想要那个粉色的！",
    ),
    # 林若溪：温柔知性女声 → XiaoxiaoNeural（温柔知性）
    "ref_female_gentle": (
        "zh-CN-XiaoxiaoNeural",
        "没事的，一切都会好起来的。我会一直在你身边，不管发生什么。",
    ),
    # 陆北辰：成熟磁性男声 → YunjianNeural（激情沉稳）
    "ref_male_deep": (
        "zh-CN-YunjianNeural",
        "这件事我自有安排，你不必担心。该来的总会来。",
    ),
    # 王妈：中年妇女 → 辽宁方言 XiaobeiNeural（幽默大嗓门，更像中年妇女）
    "ref_auntie": (
        "zh-CN-liaoning-XiaobeiNeural",
        "小姐您放心，有我老婆子在，谁也别想欺负您！我这就去收拾她！",
    ),
    # 顾明轩：年轻傲慢男声 → YunxiNeural（阳光年轻男声）
    "ref_male_arrogant": (
        "zh-CN-YunxiNeural",
        "你算什么东西？也配跟我说这种话？真是笑话！",
    ),
    # 陈甜甜：甜美做作女声 → XiaoyiNeural（活泼，已用于小七，这里用不同文本）
    #   但为了区分，改用 XiaoxiaoNeural + 不同语调文本
    "ref_female_sweet_fake": (
        "zh-CN-XiaoxiaoNeural",
        "哥哥，人家不是故意的嘛，你就别生气了好不好？人家真的很喜欢你嘛。",
    ),
    # 李氏：老年尖锐女声 → 陕西方言 XiaoniNeural（尖锐、有老太太感）
    "ref_female_sharp": (
        "zh-CN-shaanxi-XiaoniNeural",
        "哼，就凭她那种出身，也配进我们家的门？真是笑死人了！",
    ),
    # 林书远：权威男声 → YunyangNeural（专业可靠新闻主播）
    "ref_male_authoritative": (
        "zh-CN-YunyangNeural",
        "这个决定我已经做完了，不需要再讨论。按我说的去做。",
    ),
    # 陈思琪：干练女精英 → YunxiaNeural（可爱男童声作对比太奇怪）
    #   用 XiaoxiaoNeural 但不同语调文本
    "ref_female_pro": (
        "zh-CN-XiaoxiaoNeural",
        "根据我的分析，这个项目的投资回报率在百分之二十以上，建议立即推进。",
    ),
    # 叙述者：中性播音腔 → YunyangNeural（新闻主播，男女皆可的播音感）
    "ref_narrator": (
        "zh-CN-YunyangNeural",
        "夕阳的余晖洒在城市的天际线上，一切看似平静，暗涌却已在悄然酝酿。",
    ),
}

# 角色 → 参考音频文件名映射
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


async def _generate_one(name, voice, text):
    """用 Edge TTS 生成一段音频"""
    import edge_tts
    output_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    tmp_path = output_path + ".tmp.mp3"

    print(f"  Generating {name} [{voice}]...")
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        # 读取并重采样到 24000Hz
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(output_path, audio, TARGET_SR)
        duration = len(audio) / TARGET_SR
        print(f"    -> {output_path} ({duration:.1f}s, {TARGET_SR}Hz)")
    except Exception as e:
        print(f"    -> FAILED: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def generate_refs():
    print(f"Target sample rate: {TARGET_SR}Hz")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # 逐个生成（Edge TTS 限流）
    for name, (voice, text) in VOICE_PROFILES.items():
        output_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        if os.path.exists(output_path):
            print(f"  [SKIP] {name} (already exists)")
            continue
        await _generate_one(name, voice, text)

    # 打印角色映射
    print(f"\n角色→参考音频映射:")
    for char, ref in CHAR_REF_MAP.items():
        voice_name = VOICE_PROFILES[ref.replace('.wav', '')][0]
        print(f"  {char}: {ref} ({voice_name})")

    print(f"\n参考音频目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(generate_refs())
