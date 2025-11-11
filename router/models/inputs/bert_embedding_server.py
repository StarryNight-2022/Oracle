from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from typing import List, Union
import uvicorn

from router.models.modeling.modeling import Bert

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: str = None

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

class ErrorResponse(BaseModel):
    error: dict

# 设备配置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/home/ouyk/project/ICDCS/Oracle/model/Fine_Tuned'

# 加载模型和tokenizer
try:
    model = Bert(
            bert_dir=MODEL_PATH,
            device=device,
            dtype=torch.float32
            )
    model.eval()
    print(f"模型加载成功，设备: {device}")
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """生成文本的BERT embedding"""
    try:
        # 生成embedding
        with torch.no_grad():
            outputs = model(texts[0])
            # 使用[CLS] token的隐藏状态作为句子表示
            embeddings = outputs.cpu().numpy()
        
        return embeddings.tolist()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成embedding时出错: {str(e)}")

app = FastAPI(
    title="BERT Embedding API",
    description="使用BERT模型生成文本embedding的OpenAI兼容API",
    version="1.0.0"
)

@app.post("/v1/embeddings", 
          response_model=EmbeddingResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def create_embedding(request: EmbeddingRequest):
    """生成文本的embedding"""
    
    # 验证输入
    if not request.input:
        raise HTTPException(status_code=400, detail="输入不能为空")
    
    # 统一输入格式为列表
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
    
    # 生成embedding
    embeddings = get_embeddings(texts)
    
    # 构建响应数据
    data = []
    for i, embedding in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i
        })
    
    # 计算token使用量（近似值）
    total_tokens = "Not Available"
    
    response = EmbeddingResponse(
        data=data,
        model=request.model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    )
    
    return response

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": True}

@app.get("/models")
async def list_models():
    """返回支持的模型列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": "bert-embedding",
                "object": "model",
                "created": 1677610602,
                "owned_by": "local"
            }
        ]
    }

if __name__ == '__main__':
    import os
    # 获取当前文件名（不带扩展名）
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    
    uvicorn.run(
        app=f"{current_file}:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )