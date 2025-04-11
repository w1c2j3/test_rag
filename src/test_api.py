"""
简单的API测试脚本，验证自定义API连接是否正常
"""

import json
import time
from typing import Dict, Any, Optional, List, Tuple
from src.custom_api_client import CustomLLMClient

class ApiTester:
    """API测试类，封装测试逻辑"""
    
    def __init__(self, api_key: str, api_base_url: str, api_model: str):
        """
        初始化测试客户端
        
        参数:
            api_key: API密钥
            api_base_url: API基础URL
            api_model: 模型名称
        """
        # 保存API配置
        self.api_key = api_key
        self.api_url = api_base_url
        self.model_name = api_model
        
        # 创建统一的API客户端
        self.client = CustomLLMClient(
            api_url=self.api_url,
            api_key=self.api_key,
            model_name=self.model_name
        )
    
    def test_normal_call(self) -> bool:
        """
        测试普通（非流式）API调用
        
        返回:
            bool: 测试是否通过
        """
        print("测试非流式API调用...")
        try:
            start_time = time.time()
            response = self.client._call("Say 'Hello, API test is working!' in Chinese")
            end_time = time.time()
            
            print(f"响应时间: {end_time - start_time:.2f} 秒")
            print(f"API响应内容: {response}")
            print("非流式API测试成功!")
            return True
        except Exception as e:
            print(f"非流式API测试失败: {e}")
            return False
    
    def test_stream_call(self) -> bool:
        """
        测试流式API调用
        
        返回:
            bool: 测试是否通过
        """
        print("\n测试流式API调用...")
        try:
            start_time = time.time()
            full_text = ""
            
            print("流式响应开始接收:")
            print("-" * 40)
            
            # 使用generator获取流式响应
            for chunk in self.client.stream_chat("Count from 1 to 5 in Chinese, one number per line."):
                if chunk:
                    full_text += chunk
                    print(chunk, end="", flush=True)
            
            end_time = time.time()
            print("\n" + "-" * 40)
            print(f"流式响应总时间: {end_time - start_time:.2f} 秒")
            print(f"接收到的完整文本长度: {len(full_text)} 字符")
            print("流式API测试成功!")
            return True
        except Exception as e:
            print(f"流式API测试失败: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        运行所有测试
        
        返回:
            Dict[str, bool]: 测试结果字典
        """
        print("开始API测试...\n")
        
        results = {}
        
        # 测试非流式API
        results["normal_call"] = self.test_normal_call()
        
        # 测试流式API
        results["stream_call"] = self.test_stream_call()
        
        # 总结测试结果
        print("\nAPI测试结果总结:")
        print(f"- 非流式API测试: {'成功' if results['normal_call'] else '失败'}")
        print(f"- 流式API测试: {'成功' if results['stream_call'] else '失败'}")
        
        results["all_passed"] = all(results.values())
        
        if results["all_passed"]:
            print("\n所有API测试通过，可以继续使用ragas进行评估!")
        else:
            print("\n一些API测试失败，请检查API配置和连接!")
            
        return results

if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="API测试工具")
    parser.add_argument("--api-url", type=str, required=True, help="API基础URL")
    parser.add_argument("--api-key", type=str, required=True, help="API密钥")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--test-type", choices=["normal", "stream", "all"], default="all", 
                      help="要测试的API类型")
    
    args = parser.parse_args()
    
    # 创建测试器并运行测试
    tester = ApiTester(
        api_key=args.api_key,
        api_base_url=args.api_url,
        api_model=args.model
    )
    
    if args.test_type == "normal":
        tester.test_normal_call()
    elif args.test_type == "stream":
        tester.test_stream_call()
    else:
        tester.run_all_tests() 