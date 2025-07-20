import os
import yaml
import time

def load_config(config_path="rca/api_config.yaml"):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs

configs = load_config()


def OpenAI_chat_completion(messages, temperature):
    from openai import OpenAI

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=configs["API_KEY"],
        base_url=configs["API_BASE"]
    )

    # 发送请求到 OpenAI API
    response = client.chat.completions.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature,
    )

    # 获取响应内容
    response_content = response.choices[0].message.content

    # 创建或打开日志文件，并追加请求和响应
    with open("openai_chat_log.txt", "a") as log_file:
        log_file.write("Request:\n")
        for msg in messages:
            log_file.write(f"{msg}\n")
        log_file.write("Response:\n")
        log_file.write(f"{response_content}\n")
        log_file.write("----------------------------------\n")  # 使用指定的分隔符

    # 返回响应内容
    return response_content


def Google_chat_completion(messages, temperature):
    import google.generativeai as genai
    genai.configure(
        api_key=configs["API_KEY"]
    )
    genai.GenerationConfig(temperature=temperature)
    system_instruction = messages[0]["content"] if messages[0]["role"] == "system" else None
    messages = [item for item in messages if item["role"] != "system"]
    messages = [{"role": "model" if item["role"] == "assistant" else item["role"], "parts": item["content"]} for item in messages]
    history = messages[:-1]
    message = messages[-1]
    return genai.GenerativeModel(
        model_name=configs["MODEL"],
        system_instruction=system_instruction
        ).start_chat(
            history=history if history != [] else None
            ).send_message(message).text

def Anthropic_chat_completion(messages, temperature):
    import anthropic
    client = anthropic.Anthropic(
        api_key=configs["API_KEY"]
    )
    return client.messages.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature
    ).content

# for 3-rd party API which is compatible with OpenAI API (with different 'API_BASE')
def AI_chat_completion(messages, temperature):    
    from openai import OpenAI
    client = OpenAI(
        api_key=configs["API_KEY"],
        base_url=configs["API_BASE"]
    )
    return client.chat.completions.create(
        model = configs["MODEL"],
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content

def get_chat_completion(messages, temperature=0.0):

    def send_request():
        if configs["SOURCE"] == "AI":
            return AI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "OpenAI":
            return OpenAI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Google":
            return Google_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Anthropic":
            return Anthropic_chat_completion(messages, temperature)
        else:
            raise ValueError("Invalid SOURCE in api_config file.")
    
    for i in range(3):
        try:
            return send_request()
        except Exception as e:
            print(e)
            if '429' in str(e):
                print("Rate limit exceeded. Waiting for 1 second.")
                time.sleep(1)
                continue
            else:
                raise e