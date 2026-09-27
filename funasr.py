import os
import re
import time
import sys
import warnings
from pathlib import Path

# -------------------------- 强制配置hf-mirror镜像，所有模型下载自动走国内站点 --------------------------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HOME'] = str(Path(__file__).parent.resolve() / ".cache" / "huggingface")
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

# -------------------------- 配置固定路径（严格对应你要求的目录结构） --------------------------
BASE_DIR = Path(__file__).parent.resolve()
AUDIO_DIR = BASE_DIR / "v2t"
WHISPER_CACHE = BASE_DIR / ".cache" / "whisper"
SAMPLE_RATE = 16000
SUPPORTED_LANGS = {"en": "英语", "zh": "中文", "ja": "日语", "ko": "韩语"}

# -------------------------- 命令行参数配置 --------------------------
SUPPORTED_MODELS = {"small", "medium", "large"}
MODEL_MAP = {"small": "small", "medium": "medium", "large": "large-v3"}
MODEL_DESC = {
    "small": "small（~500MB，速度最快，清晰会议首选）",
    "medium": "medium（~1.5GB，通用性价比最高）",
    "large": "large-v3（~3.1GB，重噪声/影视效果最好）"
}

# -------------------------- 解析命令行参数（顺序任意，不用记位置） --------------------------
selected_model = "small"
selected_lang = None
invalid_args = []
model_count = 0
lang_count = 0
for arg in sys.argv[1:]:
    arg_lower = arg.lower().strip()
    if arg_lower in SUPPORTED_MODELS:
        selected_model = arg_lower
        model_count += 1
    elif arg_lower in SUPPORTED_LANGS:
        selected_lang = arg_lower
        lang_count += 1
    else:
        invalid_args.append(arg)

error_msg = []
if invalid_args:
    error_msg.append(f"❌ 不支持的参数：{', '.join(invalid_args)}")
if model_count > 1:
    error_msg.append("❌ 不能同时指定多个模型，只能选small/medium/large其中一个")
if lang_count > 1:
    error_msg.append("❌ 不能同时指定多个语言，只能选en/zh/ja/ko其中一个")
if error_msg:
    print("\n".join(error_msg))
    print("="*70)
    print("📖 使用说明：")
    print(f"  模型可选：small（默认） / medium / large")
    print(f"  语言可选：en / zh / ja / ko（不传则自动检测每个音频的语言）")
    print(f"  参数顺序任意，不需要记位置")
    print("\n示例：")
    print(f"  python funasr.py                # 默认small模型，逐文件自动检测语言")
    print(f"  python funasr.py medium         # medium模型，逐文件自动检测语言")
    print(f"  python funasr.py en             # small模型，所有文件按英语识别")
    print(f"  python funasr.py large ja       # large模型，所有文件按日语识别")
    sys.exit(1)

# -------------------------- 工具函数 --------------------------
def format_time(ms):
    """毫秒转SRT标准格式 HH:MM:SS,mmm"""
    ms = max(0, int(ms))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    msec = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

# -------------------------- 词级切分&翻译后处理工具函数 --------------------------
SENT_END_CHARS = (".", "!", "?", ";", "。", "！", "？", "；")
MIN_SEG_DUR = 1200   # 单段最短1.2秒
MAX_SEG_DUR = 8000   # 单段最长8秒
MIN_PAUSE = 600      # 词间停顿≥600ms视为自然断点

# -------------------------- 词级切分函数：加lang参数，自动清理日语/中文词间空格 --------------------------
# -------------------------- 词级切分函数：流式，识别一句即可打印一句 --------------------------
def group_words_to_segments(segments_gen, lang=None):
    """流式词级切分：边读边切边产出，切分逻辑与原全量版完全等价"""
    def join(ws):
        t = " ".join(x["t"] for x in ws).strip()
        if lang in ("ja", "zh"):
            t = t.replace(" ", "").replace("　", "")
        return t

    def emit(ws):
        seg_text = join(ws)
        if not seg_text:
            return None
        if (ws[-1]["e"] - ws[0]["s"]) < 500 and len(seg_text.split()) > 5:
            return None
        return (ws[0]["s"], ws[-1]["e"], seg_text)

    def words():
        for seg in segments_gen:
            if getattr(seg, "words", None):
                for w in seg.words:
                    t = (w.word or "").strip()
                    if t:
                        yield {"s": int(w.start * 1000), "e": int(w.end * 1000), "t": t}
            else:
                t = (seg.text or "").strip()
                if t:
                    yield {"s": int(seg.start * 1000), "e": int(seg.end * 1000), "t": t}

    cur_words = []
    prev = None
    for word in words():
        if prev is not None:
            cur_words.append(prev)
            cur_dur = cur_words[-1]["e"] - cur_words[0]["s"]
            gap = word["s"] - cur_words[-1]["e"]
            is_end = cur_words[-1]["t"].endswith(SENT_END_CHARS)
            need_cut = False
            if cur_dur >= MIN_SEG_DUR and (is_end or gap >= MIN_PAUSE):
                need_cut = True
            elif cur_dur >= MAX_SEG_DUR:
                max_gap = -1
                cut_idx = len(cur_words)
                for k in range(1, len(cur_words)):
                    g = cur_words[k]["s"] - cur_words[k-1]["e"]
                    if g > max_gap:
                        max_gap = g
                        cut_idx = k
                if max_gap >= 200 and cut_idx > 0 and cut_idx < len(cur_words):
                    head = cur_words[:cut_idx]
                    tail = cur_words[cut_idx:]
                    r = emit(head)
                    if r:
                        yield r
                    cur_words = tail
                else:
                    need_cut = True
            if need_cut:
                r = emit(cur_words)
                if r:
                    yield r
                cur_words = []
        prev = word
    if prev is not None:
        cur_words.append(prev)
    if cur_words:
        r = emit(cur_words)
        if r:
            yield r

# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("="*70)
    print(f"✅ 将使用识别模型：{MODEL_DESC[selected_model]}")
    if selected_lang:
        print(f"✅ 将按指定语言识别所有音频：{SUPPORTED_LANGS[selected_lang]}({selected_lang})，跳过自动检测")
    else:
        print(f"🔍 未指定统一语言，将自动逐文件检测音频语言")
    print("="*70)

    # -------------------------- 加载faster-whisper模型（int8 CPU量化，和SubtitleEdit配置完全一致） --------------------------
    from faster_whisper import WhisperModel
    local_model_dir = WHISPER_CACHE / MODEL_MAP[selected_model]
    print(f"📦 正在加载模型，优先读取本地路径：{local_model_dir}")
    if local_model_dir.is_dir() and any(local_model_dir.iterdir()):
        print(f"✅ 找到本地已缓存的模型，直接加载，无需联网")
        whisper_model = WhisperModel(
            str(local_model_dir),
            device="cpu",
            compute_type="int8"
        )
    else:
        print(f"⚠️  本地模型不存在，将自动从hf-mirror国内镜像下载完整模型")
        whisper_model = WhisperModel(
            MODEL_MAP[selected_model],
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_CACHE)
        )
    print("✅ 语音识别模型加载完成")

    # -------------------------- 保存输出文件 --------------------------
    def save_outputs(audio_path, segments, lang_code):
        stem = audio_path.stem
        out_dir = audio_path.parent
        lang_suffix = f"_{lang_code}"
        # 原语言TXT
        with open(out_dir / f"{stem}{lang_suffix}.txt", "w", encoding="utf-8") as f:
            for s, e, t in segments:
                f.write(f"[{format_time(s)} - {format_time(e)}] {t}\n")
        # 原语言SRT
        with open(out_dir / f"{stem}{lang_suffix}.srt", "w", encoding="utf-8") as f:
            for i, (s, e, t) in enumerate(segments, 1):
                f.write(f"{i}\n{format_time(s)} --> {format_time(e)}\n{t}\n\n")

    # -------------------------- 扫描v2t目录音频文件 --------------------------
    print("="*70)
    print(f"🔍 扫描音频目录：{AUDIO_DIR}")
    audio_exts = [".wav", ".mp3", ".flac", ".m4a", ".aac"]
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(AUDIO_DIR.glob(f"*{ext}"))
    audio_files = sorted(audio_files)
    if not audio_files:
        print("❌ 未找到任何音频文件，请将音频放到v2t目录下")
        sys.exit(1)
    print(f"✅ 找到{len(audio_files)}个音频文件")
    print("="*70)

    # -------------------------- 逐文件处理 --------------------------
    success = 0
    fail = 0
    skip = 0
    for f_idx, audio_path in enumerate(audio_files, 1):
        print(f"\n--- 正在处理 {f_idx}/{len(audio_files)}: {audio_path.name} ---")
        start_time = time.time()
        stem = audio_path.stem
        try:
            # 断点续跑：如果所有目标输出文件已存在，直接跳过
            if selected_lang:
                chk_lang = selected_lang
                chk_exist = (AUDIO_DIR / f"{stem}_{chk_lang}.txt").exists() and \
                               (AUDIO_DIR / f"{stem}_{chk_lang}.srt").exists()
                if chk_exist:
                    print(f"⏭️  结果文件已全部存在，跳过处理")
                    skip += 1
                    success += 1
                    continue
            else:
                # 自动检测模式下，补上日语/韩语结果文件检查，只要存在任意语种的完整结果就跳过
                chk_en = (AUDIO_DIR / f"{stem}_en.txt").exists() and (AUDIO_DIR / f"{stem}_en.srt").exists()
                chk_ja = (AUDIO_DIR / f"{stem}_ja.txt").exists() and (AUDIO_DIR / f"{stem}_ja.srt").exists()
                chk_ko = (AUDIO_DIR / f"{stem}_ko.txt").exists() and (AUDIO_DIR / f"{stem}_ko.srt").exists()
                chk_zh = (AUDIO_DIR / f"{stem}_zh.txt").exists() and (AUDIO_DIR / f"{stem}_zh.srt").exists()
                if chk_en or chk_ja or chk_ko or chk_zh:
                    print(f"⏭️  结果文件已全部存在，跳过处理")
                    skip += 1
                    success += 1
                    continue

            # 识别
            current_lang = selected_lang
            if not current_lang:
                segments_gen, info = whisper_model.transcribe(
                    audio_path,
                    language=None,
                    beam_size=5,
                    no_repeat_ngram_size=3,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    word_timestamps=True,
                    # 加上这段VAD参数，适配电话/远场弱音量场景
                    vad_parameters=dict(
                        threshold=0.3,         # 默认0.5，调低到0.3，对小音量更敏感
                        min_speech_duration_ms=200, # 默认250ms，不用改
                        min_silence_duration_ms=500, # 默认2s，调短到0.5s，不会把长语音中间的短停顿当成静音切断
                        max_speech_duration_s=12,
                    )
                )
                detect_lang = info.language
                detect_prob = info.language_probability
                current_lang = detect_lang
                print(f"🎯 自动检测到语言：{SUPPORTED_LANGS.get(detect_lang, detect_lang)}({detect_lang})，置信度：{detect_prob:.2f}")
                if detect_prob < 0.5:
                    print(f"⚠️  警告：语言检测置信度低于0.5，如结果异常请手动指定语言重新运行")
            else:
                print(f"✅ 按指定语言识别：{SUPPORTED_LANGS[current_lang]}({current_lang})")
                segments_gen, info = whisper_model.transcribe(
                    audio_path,
                    language=current_lang,
                    beam_size=5,
                    no_repeat_ngram_size=3,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    word_timestamps=True,
                    # 加上这段VAD参数，适配电话/远场弱音量场景
                    vad_parameters=dict(
                        threshold=0.3,         # 默认0.5，调低到0.3，对小音量更敏感
                        min_speech_duration_ms=200, # 默认250ms，不用改
                        min_silence_duration_ms=500, # 默认2s，调短到0.5s，不会把长语音中间的短停顿当成静音切断
                        max_speech_duration_s=12,
                    )
                )

            # 逐句打印进度：调用切分函数时传入当前语言参数，自动清理词间空格

            # 词级后处理：正确切分句子，无长空段/无硬拆句/自动过滤复读
            # 流式：边切边打印，同时收集成列表供后面保存使用
            segments = []
            for s_ms, e_ms, text in group_words_to_segments(segments_gen, current_lang):
                print(f"  [{format_time(s_ms)} - {format_time(e_ms)}] {text}", flush=True)
                segments.append((s_ms, e_ms, text))

            if not segments:
                print(f"❌ 本文件未识别到有效内容，耗时{time.time()-start_time:.1f}秒")
                fail +=1
                continue

            print(f"✅ ASR识别完成，共{len(segments)}句有效内容，开始保存结果...")
            save_outputs(audio_path, segments, current_lang)
            cost = time.time() - start_time
            print(f"✅ {audio_path.name} 处理完成，耗时{cost:.1f}秒，已生成所有输出文件")
            success +=1

        except Exception as e:
            import traceback
            print(f"❌ 处理失败：{str(e)}，耗时{time.time()-start_time:.1f}秒")
            traceback.print_exc()
            fail +=1

    # 最终统计
    print("\n" + "="*70)
    print(f"🎉 全部处理完成！成功：{success}个，跳过：{skip}个，失败：{fail}个")
    print("="*70)
    os._exit(0)
