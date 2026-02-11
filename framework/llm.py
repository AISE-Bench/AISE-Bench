from openai import OpenAI

from config import CHATGLM_API_KEY, CHATGLM_API_BASE, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, ARK_API_KEY, ARK_API_BASE, ARK_MODEL

# def llm_client(prompt, query, model=ARK_MODEL, api_key=ARK_API_KEY, base_url=ARK_API_BASE):
# def llm_client(prompt, query, model="glm-4.7", api_key=CHATGLM_API_KEY, base_url=CHATGLM_API_BASE):
def llm_client(prompt, query, model="gemini-3-pro-preview-11-2025", api_key="", base_url="https://yunwu.ai/v1"):

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        conclusion = client.chat.completions.create(
            model = model,
            messages = [
                {"role": "system", 
                "content": prompt
                },
                {"role": "user",
                "content": query
                },
            ],
            temperature=0.95,
            top_p=0.7,
        )

        return conclusion.choices[0].message.content

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    print(llm_client(
        prompt = "你是一个复读机，你会重复三遍用户问题",
        query = "你说你好",
        model="deepseek-reasoner",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE
        )
    )

    