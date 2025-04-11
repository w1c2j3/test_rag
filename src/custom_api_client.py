"""
定义一个自定义的LLM客户端，使用用户提供的API配置
"""

import json
import requests
from typing import List, Dict, Any, Optional, Union, Generator
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult

class CustomLLMClient:
    """自定义LLM客户端，使用第三方API"""
    
    def __init__(self, api_url, api_key, model_name):
        """
        初始化LLM客户端
        
        参数:
            api_url: API基础URL
            api_key: API密钥
            model_name: 模型名称
        """
        # 保存API配置
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        
        # 验证API密钥
        if not self.api_key:
            print("警告: API密钥未设置。请在main.py中设置API_KEY")
        
        # 设置API请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _prepare_request_data(self, prompt: str, stream: bool = False) -> Dict:
        """
        准备请求数据，避免代码重复
        
        参数:
            prompt: 提示文本
            stream: 是否使用流式响应
            
        返回:
            请求数据字典
        """
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
    
    def _call(self, prompt: str, **kwargs) -> str:
        """
        调用API获取非流式响应
        
        参数:
            prompt: 提示文本
            
        返回:
            API响应内容
        """
        data = self._prepare_request_data(prompt, stream=False)
        
        print(f"发送请求到 {self.api_url}...")
        print(f"使用模型: {self.model_name}")
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            print("请求成功!")
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            raise Exception(f"API调用失败: {str(e)}")
    
    def stream_chat(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        """
        调用API获取流式响应
        
        参数:
            prompt: 提示文本
            
        返回:
            流式响应生成器
        """
        data = self._prepare_request_data(prompt, stream=True)
        
        print(f"发送流式请求到 {self.api_url}...")
        print(f"使用模型: {self.model_name}")
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data_str = line[5:].strip()
                        if data_str != "[DONE]":
                            try:
                                data_json = json.loads(data_str)
                                if "choices" in data_json and len(data_json["choices"]) > 0:
                                    delta = data_json["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        full_response += content
                                        yield content
                            except json.JSONDecodeError:
                                print(f"无法解析响应: {data_str}")
            
            return full_response
        except requests.exceptions.RequestException as e:
            print(f"API流式请求失败: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            raise Exception(f"API流式调用失败: {str(e)}")


class LangchainCustomLLMWrapper(LLM):
    """为Ragas评估提供的LangChain兼容LLM包装器"""
    
    client: Optional[CustomLLMClient] = None
    
    def __init__(self, api_key, api_url, model_name, **kwargs):
        """
        初始化LLM客户端
        
        参数:
            api_key: API密钥
            api_url: API基础URL
            model_name: 模型名称
        """
        super().__init__(**kwargs)
        
        # 保存API配置
        self.api_key = api_key
        self.api_url = api_url
        self.api_base_url = api_url  # 添加兼容字段
        self.model_name = model_name
        
        # 初始化客户端
        self.client = CustomLLMClient(self.api_url, self.api_key, self.model_name)
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "custom_llm"
    
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        """LangChain LLM _call方法实现"""
        try:
            return self.client._call(prompt)
        except Exception as e:
            print(f"API调用失败: {e}")
            raise e
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数"""
        return {"model_name": self.model_name, "api_url": self.api_url}


# 使用示例
if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="API客户端测试工具")
    parser.add_argument("--api-url", type=str, required=True, help="API基础URL")
    parser.add_argument("--api-key", type=str, required=True, help="API密钥")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?", help="测试提示文本")
    parser.add_argument("--stream", action="store_true", help="使用流式响应")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = CustomLLMClient(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model
    )
    
    # 测试API调用
    if args.stream:
        print("使用流式响应:")
        for chunk in client.stream_chat(args.prompt):
            print(chunk, end="", flush=True)
        print("\n流式响应完成。")
    else:
        print("使用普通响应:")
        response = client._call(args.prompt)
        print(f"响应内容: {response}") 