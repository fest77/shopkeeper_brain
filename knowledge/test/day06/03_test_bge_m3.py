from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 加载模型
bge_m3 = BGEM3EmbeddingFunction(
    #model_name="BAAI/bge-m3",
    model_name=r"D:\ai_models\modelscope_cache\models\BAAI\bge-m3",
    #device="cuda:0",
    #use_fp16=True,
    device="cpu",
    use_fp16=False,
)

# 2. 生成嵌入
embeddings = bge_m3.encode_documents(["RS-12 数字万用表","RS-13 数字万用表"])

# 3. 提取向量
dense_vector = embeddings["dense"][0].tolist()   # List[float], 长度 1024
sparse_matrix = embeddings["sparse"]              # CSR 稀疏矩阵

# 4. 从 CSR 矩阵提取稀疏向量
# 通过 CSR 矩阵的 indptr 数组定位第一行数据的起止位置，
# 再根据该范围从 indices 和 data 中分别取出 token_id 与权重，
# 最终组装为 {token_id: weight} 的字典格式稀疏向量
"""
注释说明：这段代码的核心是从 BGE-M3 返回的 CSR 稀疏矩阵中，解析出 {token_id: weight} 字典。CSR 矩阵的三个核心数组关系如下：
indptr:   [0,  5,  12, ...]    ← 每行数据在 indices/data 中的起止下标
indices:  [3, 17, 42, 88, 103, ...]  ← 非零元素的列索引（即 token_id）
data:     [0.31, 0.52, 0.18, 0.67, 0.44, ...]  ← 非零元素的值（即权重）
"""
start_idx = sparse_matrix.indptr[0]
end_idx = sparse_matrix.indptr[1]
token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
weights = sparse_matrix.data[start_idx:end_idx].tolist()
sparse_vector = dict(zip(token_ids, weights))    # Dict[int, float]

print(start_idx)
print(end_idx)
print(token_ids)
print(weights)
print(sparse_vector)