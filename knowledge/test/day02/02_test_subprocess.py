import os
import subprocess

os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("✅ 环境变量已配置")
print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
print()

#    mineru -p input.pdf -o output_dir/ --source local  --backend  pipeline

proc = subprocess.Popen(
    args=["mineru", "-p", r"E:/doc/万用表RS-12的使用.pdf", "-o", r"E:/doc/output/", "--backend", "pipeline"],
    stdout=subprocess.PIPE,  # 捕获标准输出
    stderr=subprocess.STDOUT,  # 合并错误到标准输出
    text=True,
    encoding="utf-8",
    errors="replace",  # 遇到乱码时替换
    bufsize=1  # 行缓冲，实时输出
)

for line in proc.stdout:
    print(line.rstrip())

return_code = proc.wait()
print(return_code)
