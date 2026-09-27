import os, sys, warnings, time, re
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
BASE_DIR = Path(__file__).parent.resolve()
os.environ['HF_HOME'] = str(BASE_DIR / ".cache" / "huggingface") 
AUDIO_DIR = BASE_DIR / "v2t"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LANG_CODE = {"en":("eng_Latn","zho_Hans"), "ja":("jpn_Jpan","zho_Hans"), "ko":("kor_Hang","zho_Hans")}
TRANS_POST_EDIT = [] #[("开球会","项目启动会"),("外务省","RFA"),("直接的-烟雾麦克风","定向麦克风"),("直接的 - 烟雾麦克风","定向麦克风")]

def post_edit(t):
    for o,n in TRANS_POST_EDIT: t=t.replace(o,n)
    return t.strip()

def parse_time(s):
    # 时间格式是 HH:MM:SS,mmm，按冒号/逗号分割成4段：时、分、秒、毫秒
    h, m, s_part, ms = map(int, re.split(r'[:,]', s))
    return h * 3600000 + m * 60000 + s_part * 1000 + ms

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ['PYTHONUNBUFFERED'] = '1'
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("="*70)
    print("📦 加载NLLB翻译模型（支持英/日/韩→中文）")
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    mdl.eval()
    print("✅ 翻译模型加载完成")

    # 扫描所有原语言TXT/SRT
    files = []
    for lg in ["en","ja","ko"]:
        for p in AUDIO_DIR.glob(f"*_{lg}.txt"):
            srt_p = p.parent / p.name.replace(f"_{lg}.txt", f"_{lg}.srt")
            zh_txt = p.parent / p.name.replace(f"_{lg}.txt", "_zh.txt")
            zh_srt = p.parent / p.name.replace(f"_{lg}.txt", "_zh.srt")
            if srt_p.exists() and not (zh_txt.exists() and zh_srt.exists()):
                files.append((p, srt_p, zh_txt, zh_srt, lg))
    print(f"✅ 找到{len(files)}个待翻译原文字幕")
    ok=skip=fail=0
    for idx, (txt_p, srt_p, zh_txt_p, zh_srt_p, lg) in enumerate(files,1):
        t0=time.time()
        print(f"\n--- {idx}/{len(files)}: {txt_p.stem} ({lg}→中文) ---")
        try:
            # 读取原文本
            segs = []
            with open(txt_p, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    #m = re.match(r'$([0-9:,]+) - ([0-9:,]+)$ (.*)', line)
                    m = re.match(re.escape("[") + r'([0-9:,]+) - ([0-9:,]+)' + re.escape("]") + r' (.*)', line)
                    if m: segs.append((parse_time(m.group(1)), parse_time(m.group(2)), m.group(3)))
            if not segs: print("❌ 无有效内容"); fail+=1; continue
            src, tgt = LANG_CODE[lg]
            tok.src_lang = src
            tgt_id = tok.convert_tokens_to_ids(tgt)
            zh_texts = []
            batch_size = 8
            for i in range(0, len(segs), batch_size):
                chunk = [t for _,_,t in segs[i:i+batch_size]]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    gen = mdl.generate(**enc, forced_bos_token_id=tgt_id, num_beams=5, no_repeat_ngram_size=2, max_length=512)
                for res in tok.batch_decode(gen, skip_special_tokens=True):
                    zh = post_edit(res)
                    zh_texts.append(zh)
                    print(f"  {zh}", flush=True)
            # 保存中文TXT/SRT
            with open(zh_txt_p, "w", encoding="utf-8") as f:
                for i,(s,e,_) in enumerate(segs):
                    h1,m1,s1,ms1 = s//3600000, (s%3600000)//60000, (s%60000)//1000, s%1000
                    h2,m2,s2,ms2 = e//3600000, (e%3600000)//60000, (e%60000)//1000, e%1000
                    f.write(f"[{h1:02d}:{m1:02d}:{s1:02d},{ms1:03d} - {h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}] {zh_texts[i]}\n")
            with open(zh_srt_p, "w", encoding="utf-8") as f:
                for i,(s,e,_) in enumerate(segs,1):
                    h1,m1,s1,ms1 = s//3600000, (s%3600000)//60000, (s%60000)//1000, s%1000
                    h2,m2,s2,ms2 = e//3600000, (e%3600000)//60000, (e%60000)//1000, e%1000
                    f.write(f"{i}\n{h1:02d}:{m1:02d}:{s1:02d},{ms1:03d} --> {h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}\n{zh_texts[i-1]}\n\n")
            print(f"✅ 翻译完成，共{len(segs)}句，耗时{time.time()-t0:.1f}s"); ok+=1
        except Exception as e:
            import traceback; print(f"❌ 失败：{e}"); traceback.print_exc(); fail+=1
    print(f"\n🎉 翻译完成：成功{ok}，失败{fail}")
    os._exit(0)
