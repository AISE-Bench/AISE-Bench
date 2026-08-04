import sys
import os
import re
import langid
from openai import OpenAI

from config import CHATGLM_API_KEY, CHATGLM_API_BASE, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE

language_codes_map = {
    'aa': 'Afar',
    'ab': 'Abkhazian',
    'af': 'Afrikaans',
    'am': 'Amharic',
    'ar': 'Arabic',
    'as': 'Assamese',
    'ay': 'Aymara',
    'az': 'Azerbaijani',
    'ba': 'Bashkir',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bh': 'Bihari',
    'bi': 'Bislama',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'br': 'Breton',
    'ca': 'Catalan',
    'co': 'Corsican',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'dz': 'Dzongkha',
    'el': 'Greek',
    'en': 'English',
    'eo': 'Esperanto',
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fj': 'Fijian',
    'fo': 'Faroese',
    'fr': 'French',
    'fy': 'Western Frisian',
    'ga': 'Irish',
    'gd': 'Scottish Gaelic',
    'gl': 'Galician',
    'gn': 'Guarani',
    'gu': 'Gujarati',
    'ha': 'Hausa',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'ia': 'Interlingua',
    'id': 'Indonesian',
    'ie': 'Interlingue',
    'ik': 'Inupiaq',
    'is': 'Icelandic',
    'it': 'Italian',
    'iu': 'Inuktitut',
    'ja': 'Japanese',
    'jv': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'kl': 'Kalaallisut',
    'km': 'Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'ks': 'Kashmiri',
    'ku': 'Kurdish',
    'ky': 'Kirghiz',
    'la': 'Latin',
    'ln': 'Lingala',
    'lo': 'Lao',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mg': 'Malagasy',
    'mi': 'Maori',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mo': 'Moldavian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'mt': 'Maltese',
    'my': 'Burmese',
    'na': 'Nauru',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'oc': 'Occitan',
    'om': 'Oromo',
    'or': 'Oriya',
    'pa': 'Punjabi',
    'pl': 'Polish',
    'ps': 'Pashto',
    'pt': 'Portuguese',
    'qu': 'Quechua',
    'rm': 'Romansh',
    'rn': 'Kirundi',
    'ro': 'Romanian',
    'ru': 'Russian',
    'rw': 'Kinyarwanda',
    'sa': 'Sanskrit',
    'sd': 'Sindhi',
    'sg': 'Sango',
    'sh': 'Serbo-Croatian',
    'si': 'Sinhalese',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sm': 'Samoan',
    'sn': 'Shona',
    'so': 'Somali',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'ss': 'Swati',
    'st': 'Sesotho',
    'su': 'Sudanese',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'te': 'Telugu',
    'tg': 'Tajik',
    'th': 'Thai',
    'ti': 'Tigrinya',
    'tk': 'Turkmen',
    'tl': 'Tagalog',
    'tn': 'Tswana',
    'to': 'Tonga',
    'tr': 'Turkish',
    'ts': 'Tsonga',
    'tt': 'Tatar',
    'tw': 'Twi',
    'ug': 'Uighur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'vo': 'Volapuk',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'za': 'Zhuang',
    'zh': 'Chinese',
    'zu': 'Zulu'
}

# 字符集
language_codes_char_set = {
    'ja': r'[\u3040-\u309F\u30A0-\u30FF]+',    # 日文假名
    'co': r'[\uac00-\ud7af]+',                 # 韩文
    'zh': r'[\u4e00-\u9fa5]+',                 # 中文
}

# 规则默认映射
language_codes_rule_reflect = {
    'ja': 'zh',
    'co': 'zh',
    'zh': 'zh'
}

def detect_language(text):
    # 过滤所有括号
    text = filter_brackets_content(text)

    # 模型判断
    language_code, confidence = langid.classify(text)

    # 规则判断
    for language, reflect in language_codes_rule_reflect.items():
        if language_code == language and not contains_character_set(text, language_codes_char_set[language]):
            language_code = reflect

    if full_ascii(text):
        language_code = 'en'

    # 含有中文的一律定向到中文（非日韩）
    if language_code not in ['ja', 'co', 'zh']:
        if contains_character_set(text, language_codes_char_set['zh']):
            language_code = 'zh'

    return language_code

def filter_brackets_content(text):
    # 正则表达式匹配小中大括号以及中英文书名号内的内容
    patterns = [
        r'\(.*?\)',
        r'\[.*?\]',
        r'\{.*?\}',
        r'《.*?》',
        r'‘.*?’',
        r'“.*?”',
        r'〈.*?〉',
        r'【.*?】',
        r'「.*?」',
        r'\".*?\"',
        r'\'.*?\'',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text)

    return text

def contains_character_set(text, character_set):
    # 正则表达式匹配平假名和片假名
    kana_pattern = re.compile(character_set)
    # 检查字符串中是否含有匹配的假名
    return kana_pattern.search(text)

def full_ascii(s: str) -> bool:
    # 计算字符串中英文字母的数量
    alpha_count = sum(c.isascii() for c in s)
    # print(alpha_count)
    # 计算字符串的总长度
    total_length = len(s)
    # print(total_length)
    # 判断英文字母数量是否大于总长度的50%
    return alpha_count == total_length

def translate_to_english(text):

    client = OpenAI(
        api_key = DEEPSEEK_API_KEY,
        base_url = DEEPSEEK_API_BASE
    )

    translator = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": "You are a professional translation assistant responsible for translating sentences from various languages into English. Your work principles are as follows: 1. Do not add characters arbitrarily, such as adding a hyphen in the middle of a person’s name. 2. Just output the translated English sentence and make sure not to answer any notes. 3. Do not retain any source language, including keywords. 4. If a Chinese name appears, the family name comes after, and the given name comes first!!"},
            {"role": "user", "content": text}
        ],
        top_p=0.7,
        temperature=0.9       
    )

    return translator.choices[0].message.content


def translate_to_chinese(text):

    client = OpenAI(
        api_key = DEEPSEEK_API_KEY,
        base_url = DEEPSEEK_API_BASE
    )

    translator = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": "你是一个专业的翻译助手，负责将各种语言的句子翻译成中文。你的工作原则如下：1. 不要随意添加字符，例如在人的名字中间添加连字符。2. 只输出翻译后的中文句子，并确保不要回答任何备注"},
            {"role": "user", "content": text}
        ],
        top_p=0.7,
        temperature=0.9       
    )

    return translator.choices[0].message.content

if __name__ == "__main__":
    print(translate_to_chinese("nlp"))
    print(detect_language("こんにちは、世界上的人"))