import json
import os
import subprocess
import sys
import time

# ================= 配置区域 =================
INPUT_DIR = "v"        # 输入目录（放影音文件）
OUTPUT_DIR = "v2t"     # 输出目录（存放转换后的音频）
AUDIO_FORMAT = "wav"   # 目标音频格式：标准 16kHz 单声道 16bit PCM WAV
PROGRESS_BAR_LEN = 35  # 进度条长度
# ===========================================

# 常见影音文件扩展名
VIDEO_AUDIO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv', '.mpeg', '.mpg', '.3gp',
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.ts', '.rmvb'
}

# Windows 下隐藏子进程黑窗口；非 Windows 传 0（条件表达式只求值选中的分支，因此非 Windows 不会触发 AttributeError）
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def ensure_dir(directory):
    """目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[信息] 创建输出目录: {directory}")

def get_audio_channel_info(file_path):
    """用 ffprobe 读取音频声道数与声道布局，返回 (声道数量, 声道布局)；探测失败按双声道处理"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=channels,channel_layout',
        '-print_format', 'json',
        file_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding='utf-8',
            errors='ignore',
            creationflags=NO_WINDOW
        )
        streams = json.loads(result.stdout).get('streams', [])
        if not streams:
            return 2, 'stereo'  # 检测失败默认按双声道处理
        channels = streams[0].get('channels', 2)
        channel_layout = streams[0].get('channel_layout', 'stereo' if channels == 2 else 'unknown')
        return channels, channel_layout
    except Exception:
        return 2, 'stereo'

def get_total_duration(file_path):
    """用 ffprobe 读取媒体总时长（秒），失败返回 0"""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding='utf-8',
            errors='ignore',
            creationflags=NO_WINDOW
        )
        return float(json.loads(result.stdout).get('format', {}).get('duration', 0.0))
    except Exception:
        return 0.0


def should_extract_center(channels, channel_layout):
    """
    判断是否需要单独提取中置声道。
    仅当「6 声道」且「布局为 5.1（含 5.1(side)）」时，人声集中在中置声道，单独提取可明显提升识别率；
    其余情况（单声道 / 立体声 / 其他多声道）一律合并为单声道。
    命名与实际处理共用本函数，避免两者判定不一致。
    """
    return channels == 6 and ('5.1' in channel_layout or 'side' in channel_layout)


def time_format(seconds):
    """秒数 -> MM:SS 或 HH:MM:SS；非法值返回 --:--"""
    if seconds < 0 or seconds == float('inf'):
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def print_progress_bar(current_sec, total_sec, speed, bar_len=PROGRESS_BAR_LEN):
    """在终端同一行动态刷新进度条"""
    if total_sec <= 0:
        # 无法获取总时长时，仅显示已处理时间
        progress_text = f"  进度: {time_format(current_sec)} / 未知时长  速度: {speed:.1f}x"
        sys.stdout.write(f"\r{progress_text:<100}")
        sys.stdout.flush()
        return

    percent = min(current_sec / total_sec, 1.0)
    filled = int(round(bar_len * percent))
    bar = '█' * filled + '░' * (bar_len - filled)
    pct_str = f"{percent * 100:5.1f}%"
    
    elapsed = time_format(current_sec)
    total = time_format(total_sec)
    
    if speed > 0:
        eta = (total_sec - current_sec) / speed
        eta_str = f"剩余 {time_format(eta)}"
    else:
        eta_str = "剩余 --:--"
    
    line = f"  [{bar}] {pct_str}  {elapsed}/{total}  {speed:.1f}x  {eta_str}"
    sys.stdout.write(f"\r{line:<110}")
    sys.stdout.flush()

def convert_with_progress(input_path, output_path, channels, channel_layout):
    """
    调用 ffmpeg 转换并实时显示进度。
    5.1 声道：用 pan 滤镜只取中置声道（c2=FC），人声更干净；
    其他声道：直接合并（-ac 1）为单声道。
    输出统一为 16kHz / 16bit / 单声道 PCM WAV。
    返回 (是否成功, 耗时秒数, 处理模式)
    """
    total_duration = get_total_duration(input_path)
    extract_center = should_extract_center(channels, channel_layout)
    process_mode = "提取中置人声" if extract_center else "合并单声道"

    cmd = ['ffmpeg', '-i', input_path, '-vn', '-sn', '-dn']
    if extract_center:
        cmd += ['-af', 'pan=mono|c0=c2']
    cmd += [
        '-ar', '16000',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        '-y',
        '-progress', 'pipe:1',
        '-nostats',
        '-loglevel', 'error',
        output_path,
    ]
    start_time = time.time()
    current_sec = 0.0
    last_speed = 0.0
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            creationflags=NO_WINDOW
        )
        # 逐行读取 -progress 的输出。注意 ffmpeg 的 out_time_ms 实际单位是「微秒」，故除以 1_000_000
        for line in proc.stdout:
            line = line.strip()
            
            if line.startswith('out_time_ms='):
                try:
                    current_sec = int(line.split('=', 1)[1]) / 1_000_000
                except (ValueError, IndexError):
                    pass
                if current_sec > 0:
                    print_progress_bar(current_sec, total_duration, last_speed)

            elif line.startswith('speed='):
                try:
                    val = line.split('=', 1)[1].strip().rstrip('x')
                    last_speed = float(val) if val else 0.0
                except (ValueError, IndexError):
                    pass
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        elapsed = time.time() - start_time
        
        if proc.returncode != 0:
            print()
            error_msg = stderr.strip().split('\n')[-1] if stderr.strip() else "未知错误"
            print(f"[错误] FFmpeg 返回码: {proc.returncode}")
            print(f"       信息: {error_msg[:200]}")
            return False, elapsed, process_mode
        
        # 完成时刷新到100%
        if total_duration > 0:
            print_progress_bar(total_duration, total_duration, last_speed)
        print()  # 换行
        return True, elapsed, process_mode
        
    except FileNotFoundError:
        print("\n[严重错误] 未找到 ffmpeg！请确保已安装并添加到系统 PATH 中。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 发生异常: {str(e)}")
        return False, time.time() - start_time, process_mode

def get_file_size_str(file_path):
    """把文件字节数格式化成易读字符串（B / KB / MB / GB）"""
    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        return "未知大小"

    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def main():
    print("=" * 60)
    print("  批量影音音频提取工具 (ffmpeg)")
    print("  输入 -> v/   输出 -> v2t/   格式 -> 16kHz单声道WAV")
    print("  规则：5.1声道自动提取中置人声，其余自动合并为单声道")
    print("=" * 60)
    
    if not os.path.exists(INPUT_DIR):
        print(f"[错误] 输入目录 '{INPUT_DIR}' 不存在！请先创建目录并放入影音文件。")
        return
    
    ensure_dir(OUTPUT_DIR)
    
    # 递归收集所有支持格式的影音文件
    files_to_process = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_AUDIO_EXTENSIONS:
                files_to_process.append(os.path.join(root, f))
    
    if not files_to_process:
        print(f"[提示] 在 '{INPUT_DIR}' 目录中未找到支持的影音文件。")
        return
    
    total_files = len(files_to_process)
    print(f"[信息] 共找到 {total_files} 个文件待处理")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_start = time.time()
    
    for idx, input_path in enumerate(files_to_process, 1):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # 先探测声道信息，据此决定输出命名与处理方式（命名与实际处理共用同一判定，保证一致）
        channels, channel_layout = get_audio_channel_info(input_path)
        if should_extract_center(channels, channel_layout):
            output_name = f"{base_name}-ch2_FC_中置人声.{AUDIO_FORMAT}"
        else:
            output_name = f"{base_name}.{AUDIO_FORMAT}"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"[{idx}/{total_files}] {os.path.basename(input_path)} ({get_file_size_str(input_path)})")
        print(f"  声道信息: {channels}声道 ({channel_layout})")
        print(f"  输出 -> {output_name}")

        # 已存在同名文件时直接覆盖（ffmpeg 已带 -y 参数）
        if os.path.exists(output_path):
            print("  [提示] 文件已存在，将覆盖...")

        
        success, elapsed, process_mode = convert_with_progress(input_path, output_path, channels, channel_layout)
        
        if success:
            out_size = get_file_size_str(output_path)
            print(f"  ✔ 完成  处理模式: {process_mode}  耗时: {elapsed:.1f}秒  输出大小: {out_size}")
            success_count += 1
        else:
            print(f"  ✘ 失败")
            fail_count += 1
        print()
    
    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"  处理全部完成！")
    print(f"  总计: {total_files} 个文件")
    print(f"  成功: {success_count}  失败: {fail_count}")
    print(f"  总耗时: {time_format(total_elapsed)}")
    print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  输出格式: 16000Hz 16bit 单声道 PCM WAV (标准ASR格式)")
    print("=" * 60)

if __name__ == '__main__':
    main()
