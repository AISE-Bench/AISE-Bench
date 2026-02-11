CHATGLM_API_KEY = "" 
CHATGLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

# Gemini 配置（通过 yunwu.ai OpenAI 兼容接口）
GEMINI_API_KEY = ""
GEMINI_API_BASE = "https://yunwu.ai/v1"
GEMINI_MODEL = "gemini-3-pro-preview-11-2025"

# GPT 配置（走云雾 API，供 from_plan_to_result_glm 等使用）
OPENAI_API_KEY = GEMINI_API_KEY
OPENAI_API_BASE = "https://yunwu.ai/v1"
GPT_MODEL = "gpt-5.2"

# Claude 配置（走云雾 API，供 from_plan_to_result_claude 使用）
CLAUDE_API_KEY = GEMINI_API_KEY
CLAUDE_API_BASE = "https://yunwu.ai/v1"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # 云雾支持的 Claude 模型名，需在控制台确认该模型有可用渠道

# Qwen 配置（走云雾 API，供 from_plan_to_result_qwen 使用）
QWEN_API_KEY = GEMINI_API_KEY
QWEN_API_BASE = "https://yunwu.ai/v1"
QWEN_MODEL = "qwen3-235b-a22b"  # 云雾支持的 Qwen 模型名，按控制台实际名称修改

# API
API_TOKEN = ""
API_CONFIG = {
    "search_paper_id": {
        "endpoint": "/gateway/api/v3/paper/search/paper/SearchPro",
        "method": "POST"
    },
    "search_paper_detail": {
        "endpoint": "/gateway/api/v3/paper/detail/batch/order",
        "method": "POST"
    },
    "search_venue_id": {
        "endpoint": "/gateway/api/v3/venue/search/venue/SearchPro",
        "method": "POST"
    },
    "search_venue_detail": {
        "endpoint": "/gateway/api/v3/venue/detail/batch",
        "method": "POST"
    },
    "search_author_id": {
        "endpoint": "/gateway/api/v3/person/search/aminer",
        "method": "POST"
    },
    "search_author_detail": {
        "endpoint": "/gateway/api/v3/person/detail/batch",
        "method": "POST"
    },
    "search_org_id": {
        "endpoint": "/gateway/open_platform/api/organization/search",
        "method": "POST"
    },
    "search_org_detail": {
        "endpoint": "/gateway/api/v3/organization/detail/batch",
        "method": "POST"
    },
}


GOOGLE_API_KEY = ""
BIGDATA_API_KEY = ""
BIGDATA_API_BASE = "https://api.chatglm.cn/v1"

ARK_API_KEY = ""
ARK_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "deepseek-v3-2-251201"

DEEPSEEK_API_KEY = "" # 官网
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # 可选: deepseek-reasoner 等