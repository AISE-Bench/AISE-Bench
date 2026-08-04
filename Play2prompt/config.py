# 模型
CHATGLM_API_KEY = ""  
CHATGLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

DEEPSEEK_API_KEY = "" 
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

API_KEY = "" 
API_BASE = "https://api-gateway.glm.ai/v1"

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