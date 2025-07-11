import json
import requests
import gradio as gr
from typing import Iterator
from datetime import datetime
import os
import tiktoken
from pathlib import Path

# 初始化tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# 配置常量
MAX_TOKENS = 4000
MAX_HISTORY_ITEMS = 20
MAX_CONVERSATION_LENGTH = 10000

# 确保数据目录存在
os.makedirs("data/conversations", exist_ok=True)
os.makedirs("data/training_data", exist_ok=True)

# --- 新增的缺失函数 ---
def load_profile():
    """加载用户配置文件"""
    try:
        with open('profile.json', 'r', encoding='utf-8') as f:
            profile = json.load(f)
            return profile['my_profile']
    except FileNotFoundError:
        print("错误: profile.json 文件未找到")
        # 创建默认配置文件
        default_profile = {
            "my_profile": {
                "name": "用户",
                "age": 20,
                "profession": "未设置",
                "interests": ["未设置"],
                "memory": []
            }
        }
        with open('profile.json', 'w', encoding='utf-8') as f:
            json.dump(default_profile, f, ensure_ascii=False, indent=2)
        return default_profile['my_profile']
    except Exception as e:
        print(f"加载配置文件错误: {str(e)}")
        return None

def save_training_data(message: str, profile: dict):
    """保存训练数据"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp": timestamp,
            "message": message,
            "profile": profile
        }
        filename = f"data/training_data/training_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存训练数据失败: {str(e)}")
# --------------------

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def trim_conversation(conversation: list, max_tokens: int) -> list:
    total_tokens = 0
    trimmed = []
    
    for item in reversed(conversation):
        user_msg, bot_msg = item
        item_tokens = count_tokens(user_msg) + count_tokens(bot_msg)
        
        if total_tokens + item_tokens > max_tokens:
            break
            
        trimmed.insert(0, item)
        total_tokens += item_tokens
    
    return trimmed

def save_conversation(conversation: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"data/conversations/conversation_{timestamp}"
    
    conversation_str = json.dumps(conversation, ensure_ascii=False)
    if len(conversation_str) > MAX_CONVERSATION_LENGTH:
        parts = len(conversation_str) // MAX_CONVERSATION_LENGTH + 1
        for i in range(parts):
            part = conversation_str[i*MAX_CONVERSATION_LENGTH : (i+1)*MAX_CONVERSATION_LENGTH]
            filename = f"{base_filename}_part{i+1}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(part)
    else:
        filename = f"{base_filename}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(conversation_str)

def prepare_api_messages(conversation: list, profile: dict, new_message: str) -> list:
    system_prompt = {
        "role": "system",
        "content": f"""你正在与{profile['name']}对话:
        年龄: {profile['age']}
        职业: {profile['profession']}
        兴趣: {', '.join(profile['interests'])}"""
    }
    
    messages = [system_prompt]
    tokens_used = count_tokens(system_prompt["content"])
    
    for user_msg, bot_msg in conversation[-MAX_HISTORY_ITEMS:]:
        user_content = {"role": "user", "content": user_msg}
        bot_content = {"role": "assistant", "content": bot_msg}
        
        new_tokens = count_tokens(user_msg) + count_tokens(bot_msg)
        if tokens_used + new_tokens > MAX_TOKENS:
            break
            
        messages.extend([user_content, bot_content])
        tokens_used += new_tokens
    
    if count_tokens(new_message) + tokens_used < MAX_TOKENS:
        messages.append({"role": "user", "content": new_message})
    
    return messages

def call_deepseek_api_stream(prompt: str, conversation: list, profile: dict) -> Iterator[str]:
    api_url = "https://api.deepseek.com/v1/chat/completions"
    api_key = "sk-********************************"  # 请替换为您的API密钥
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = prepare_api_messages(conversation, profile, prompt)
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,
        "max_tokens": min(2000, MAX_TOKENS - count_tokens(str(messages)))
    }
    
    try:
        with requests.post(api_url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            json_data = decoded_line[5:].strip()
                            if json_data != "[DONE]":
                                try:
                                    chunk = json.loads(json_data)
                                    if "choices" in chunk and chunk["choices"]:
                                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    pass
            else:
                yield f"[API 错误] 状态码: {response.status_code}"
    except Exception as e:
        yield f"[连接错误] {str(e)}"

def respond(message: str, chat_history: list, profile: dict):
    if not message.strip():
        yield chat_history
        return
    
    try:
        save_training_data(message, profile)
        trimmed_history = trim_conversation(chat_history, MAX_TOKENS // 2)
        
        bot_message = ""
        for chunk in call_deepseek_api_stream(message, trimmed_history, profile):
            bot_message += chunk
            yield trimmed_history + [(message, bot_message)]
        
        full_conversation = trimmed_history + [(message, bot_message)]
        save_conversation(full_conversation)
        yield full_conversation
    except Exception as e:
        print(f"对话出错: {str(e)}")
        yield chat_history + [(message, f"发生错误: {str(e)}")]

def create_interface(profile: dict):
    with gr.Blocks(title="DeepSeek 聊天助手", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 用户资料")
                gr.Markdown(f"""
                **姓名**: {profile['name']}  
                **年龄**: {profile['age']}  
                **职业**: {profile['profession']}  
                **兴趣**: {', '.join(profile['interests'])}
                """)
        
        chatbot = gr.Chatbot(
            height=500,
            scale=2,
            bubble_full_width=True,
            avatar_images=(
                "https://avatars.githubusercontent.com/u/14957082?s=200&v=4",
                "./icon.png"
            ),
            show_copy_button=True,
            layout="panel"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="输入你的消息...",
                show_label=False,
                container=False,
                autofocus=True,
                scale=4
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)
            clear = gr.Button("清空", scale=1)
        
        msg.submit(
            respond,
            [msg, chatbot, gr.State(profile)],
            chatbot,
            show_progress="hidden"
        ).then(
            lambda: "", None, msg
        )
        
        submit_btn.click(
            respond,
            [msg, chatbot, gr.State(profile)],
            chatbot,
            show_progress="hidden"
        ).then(
            lambda: "", None, msg
        )
        
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return demo

def ensure_dir(path):
    """确保目录存在且有写入权限"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o755)
    except Exception as e:
        print(f"无法创建目录 {path}: {str(e)}")
        # 备用目录（如/tmp）
        temp_path = f"/tmp/ai_conversations/{os.getlogin()}"
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        return temp_path
    return path

# 在保存文件前调用
conversation_dir = ensure_dir("data/conversations")

if __name__ == "__main__":
    profile = load_profile()
    if profile:
        demo = create_interface(profile)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            favicon_path=None
        )
    else:
        print("无法加载用户配置，请检查profile.json文件")
