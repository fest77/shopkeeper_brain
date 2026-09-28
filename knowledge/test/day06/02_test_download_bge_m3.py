from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-m3",
                              local_dir=r"D:\ai_models312\modelscope_cache\models\BAAI\bge-m3")
print(local_dir)


#  4.27G   modelscope魔塔下载

#
# Python 3.10.0 下载Bge-m3模型 出现下面错误。 大于  Python 3.10.1 版本已经解决该问题。
#
# Traceback (most recent call last):
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\my_test\day06\02_test_download_bge_m3.py", line 3, in <module>
#     local_dir = snapshot_download(model_id="BAAI/bge-m3",
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope\hub\snapshot_download.py", line 93, in snapshot_download
#     return _compat_snapshot_download(
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\compat\snapshot_download.py", line 73, in snapshot_download
#     api = HubApi(token=token, endpoint=endpoint)
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\api.py", line 147, in __init__
#     base = config or get_default_config()
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\config.py", line 315, in get_default_config
#     _default_config = HubConfig()
#   File "<string>", line 7, in __init__
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\config.py", line 107, in __post_init__
#     self.token = self.load_token()
#   File "D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\lib\site-packages\modelscope_hub\config.py", line 202, in load_token
#     if self._logged_out:
# AttributeError: 'HubConfig' object has no attribute '_logged_out'
