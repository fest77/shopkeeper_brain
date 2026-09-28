#test_bge_rerank_model.py
from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large",
              local_dir="D:\\ai_models\\modelscope_cache\\models\\BAAI\\bge-reranker-large")

print(local_dir)


"""

Traceback (most recent call last): File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\test\day11-rrf-rerank\01_test_download_reranker.py", 
line 4, in <module> local_dir = snapshot_download(model_id="BAAI/bge-reranker-large", File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope\hub\snapshot_download.py", 
line 93, in snapshot_download return _compat_snapshot_download( File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\compat\snapshot_download.py", 
line 73, in snapshot_download api = HubApi(token=token, endpoint=endpoint) File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\api.py", 
line 147, in __init__ base = config or get_default_config() File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\config.py", 
line 315, in get_default_config _default_config = HubConfig() File "<string>", line 7, in __init__ File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\config.py", 
line 107, in __post_init__ self.token = self.load_token() File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\config.py", 
line 202, in load_token if self._logged_out: AttributeError: 'HubConfig' object has no attribute '_logged_out'


原因分析
这个错误我们之前遇到过，是 Python 3.10.0 的 @dataclass(slots=True) 的一个已知 bug。
具体原因：
modelscope_hub 库的 HubConfig 类使用了 @dataclass(slots=True) 装饰器
在 Python 3.10.0 中，__post_init__ 方法会在所有 slot 属性赋值之前被调用
所以当 __post_init__ → load_token() 中访问 self._logged_out 时，该 slot 属性还没有被初始化到实例上，导致 AttributeError
注意：这个 bug 在 Python 3.10.1+ 的后续版本中已修复，但你当前使用的是 3.10.0。
"""