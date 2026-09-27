import os
import subprocess
import sys
import re
import time

# ================= 配置区域 =================
INPUT_DIR = "v"        # 输入目录
OUTPUT_DIR = "v2t"     # 输出目录
AUDIO_FORMAT = "m4a"   # 目标音频格式
AUDIO_BITRATE = "256k" # 音频比特率 (128k/192k/256k/320k)
PROGRESS_BAR_LEN = 35  # 进度条长度
# ===========================================

# 常见影音文件扩展名
VIDEO_AUDIO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv', '.mpeg', '.mpg', '.3gp',
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.ts', '.rmvb'
}

def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[信息] 创建输出目录: {directory}")

def get_total_duration(file_path):
    """使用 ffprobe 获取媒体文件总时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
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
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        import json
        data = json.loads(result.stdout)
        return float(data.get('format', {}).get('duration', 0.0))
    except Exception:
        return 0.0

def time_format(seconds):
    """将秒数格式化为 MM:SS 或 HH:MM:SS"""
    if seconds < 0 or seconds == float('inf'):
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def print_progress_bar(current_sec, total_sec, speed, bar_len=PROGRESS_BAR_LEN):
    """在终端同一行动态打印进度条"""
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

def convert_with_progress(input_path, output_path):
    """
    执行 ffmpeg 转换并实时显示进度
    返回: (是否成功, 耗时秒数)
    """
    total_duration = get_total_duration(input_path)
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vn', '-sn', '-dn',
        '-acodec', 'aac',
        '-b:a', AUDIO_BITRATE,
        '-y',
        '-progress', 'pipe:1',
        '-nostats',
        '-loglevel', 'error',
        output_path
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
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        for line in proc.stdout:
            line = line.strip()
            
            if line.startswith('out_time_ms='):
                try:
                    current_sec = int(line.split('=', 1)[1]) / 1_000_000
                except (ValueError, IndexError):
                    pass
            
            elif line.startswith('speed='):
                try:
                    val = line.split('=', 1)[1].strip().rstrip('x')
                    last_speed = float(val) if val else 0.0
                except (ValueError, IndexError):
                    pass
            
            elif line == 'progress=end':
                # 处理完成
                if total_duration > 0:
                    print_progress_bar(total_duration, total_duration, last_speed)
                break
            
            # 刷新进度条
            if current_sec > 0:
                print_progress_bar(current_sec, total_duration, last_speed)
        
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        elapsed = time.time() - start_time
        
        if proc.returncode != 0:
            print()
            error_msg = stderr.strip().split('\n')[-1] if stderr.strip() else "未知错误"
            print(f"[错误] FFmpeg 返回码: {proc.returncode}")
            print(f"       信息: {error_msg[:200]}")
            return False, elapsed
        
        # 完成时刷新到100%
        if total_duration > 0:
            print_progress_bar(total_duration, total_duration, last_speed)
        print()  # 换行
        return True, elapsed
        
    except FileNotFoundError:
        print("\n[严重错误] 未找到 ffmpeg！请确保已安装并添加到系统 PATH 中。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 发生异常: {str(e)}")
        return False, time.time() - start_time

def get_file_size_str(file_path):
    """获取文件大小的易读字符串"""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
    except OSError:
        return "未知大小"

def main():
    print("=" * 60)
    print("  批量影音文件提取音频工具 (ffmpeg)")
    print("  输入 -> v/   输出 -> v2t/   格式 -> m4a")
    print("=" * 60)
    
    if not os.path.exists(INPUT_DIR):
        print(f"[错误] 输入目录 '{INPUT_DIR}' 不存在！请先创建目录并放入影音文件。")
        return
    
    ensure_dir(OUTPUT_DIR)
    
    # 收集所有待处理文件
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
        output_name = f"{base_name}.{AUDIO_FORMAT}"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        in_size = get_file_size_str(input_path)
        
        print(f"[{idx}/{total_files}] {os.path.basename(input_path)} ({in_size})")
        print(f"  输出 -> {output_name}")
        
        # 如果已存在同名输出文件，可选择跳过或覆盖（这里默认覆盖，加了-y参数）
        if os.path.exists(output_path):
            print(f"  [提示] 文件已存在，将覆盖...")
        
        success, elapsed = convert_with_progress(input_path, output_path)
        
        if success:
            out_size = get_file_size_str(output_path)
            print(f"  ✔ 完成  耗时: {elapsed:.1f}秒  输出大小: {out_size}")
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
    print("=" * 60)

if __name__ == '__main__':
    main()
