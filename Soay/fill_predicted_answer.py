import json
import re

# 读取0131-soay.json文件，提取id和answer
def extract_answers(soay_file):
    with open(soay_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    answers = {}
    for item in data:
        qid = item['id']
        summary = item.get('summary', '')
        
        # 从summary字段中提取answer
        if summary:
            # 尝试匹配JSON格式的answer
            match = re.search(r'"answer":\s*"([^"]*)"', summary, re.DOTALL)
            if match:
                answer = match.group(1)
                # 去除转义字符
                answer = answer.replace('\\n', '\n').replace('\\"', '"')
                answers[qid] = answer
            else:
                # 如果没有找到JSON格式的answer，尝试直接提取文本
                answers[qid] = summary
        else:
            answers[qid] = ''
    
    return answers

# 读取gemini-soay_eval_results.jsonl文件，并填充predicted_answer字段
def fill_predicted_answer(input_file, output_file, answers_dict):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            qid = data['id']
            
            # 查找对应的answer
            if qid in answers_dict:
                data['predicted_answer'] = answers_dict[qid]
            else:
                data['predicted_answer'] = ''
            
            # 写入更新后的数据
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 提取答案
    answers = extract_answers('0131-soay.json')
    print(f"成功提取 {len(answers)} 个答案")
    
    # 填充predicted_answer字段
    input_file = 'gemini-soay_eval_results.jsonl'
    output_file = 'gemini-soay_eval_results_filled.jsonl'
    fill_predicted_answer(input_file, output_file, answers)
    print(f"成功填充 {output_file}")
    
    # 同样处理deepseek和qwen3的文件
    input_file = 'deepseek-soay_eval_results.jsonl'
    output_file = 'deepseek-soay_eval_results_filled.jsonl'
    fill_predicted_answer(input_file, output_file, answers)
    print(f"成功填充 {output_file}")
    
    input_file = 'qwen3-235b-a22b-soay_eval_results.jsonl'
    output_file = 'qwen3-235b-a22b-soay_eval_results_filled.jsonl'
    fill_predicted_answer(input_file, output_file, answers)
    print(f"成功填充 {output_file}")