import dashscope
from http import HTTPStatus
import argparse
import json
import time
import os
import re
from tqdm import tqdm

# ====================================================================
# --- 完整的指令模板 ---
# 这个模板教Qwen如何扮演一个客观的裁判。
# ====================================================================
template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''


def main(args):
    """
    主函数，执行完整的评估流程。
    如果评估文件已存在，则直接分析；否则，创建目录并执行API调用。
    """
    # --- 第一步：检查评估文件是否已存在 ---
    if os.path.exists(args.evaluation):
        print(f"评估文件 '{args.evaluation}' 已存在。")
        print("将直接加载此文件进行分析，跳过API调用。")
        try:
            with open(args.evaluation, 'r', encoding='utf-8') as f:
                raw_responses = json.load(f)
            # 直接跳转到分析函数
            analyze_results(raw_responses, args)
            return  # 分析完成后直接退出程序
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告：无法读取或解析已存在的评估文件 '{args.evaluation}'。错误: {e}")
            print("将继续执行API调用流程来重新生成文件。")

    # --- 如果文件不存在，则执行下面的API调用流程 ---

    print("评估文件不存在，开始执行API调用流程...")

    # 第二步：确保输出目录存在
    eval_dir = os.path.dirname(args.evaluation)
    if eval_dir:
        os.makedirs(eval_dir, exist_ok=True)
        print(f"确保目录 '{eval_dir}' 已创建。")

    # 第三步：配置API Key
    if args.api_key:
        dashscope.api_key = args.api_key
    else:
        try:
            with open("dashscope_apikey.txt", "r") as f:
                dashscope.api_key = f.read().strip()
        except FileNotFoundError:
            dashscope.api_key = None

    if not dashscope.api_key:
        raise ValueError("API Key 未配置。请创建 dashscope_apikey.txt 文件或使用 --api-key 参数提供。")

    # 第四步：加载需要评估的回答文件
    try:
        with open(args.response, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到响应文件 '{args.response}'。请检查路径。")
        return
    except json.JSONDecodeError:
        print(f"错误：响应文件 '{args.response}' 不是一个有效的JSON文件。")
        return

    # 第五步：执行API调用循环（带内容审查失败跳过 & 有限重试）
    raw_responses = []
    print(f"开始使用模型 '{args.qwen_model}' 进行新评估...")
    for i, record in enumerate(tqdm(records, desc="正在评估")):
        required_keys = ['image_content', 'question', 'gt_answer', 'model_answer']
        if not all(key in record for key in required_keys):
            print(f"\n警告：第 {i} 个样本缺少必要的键，跳过。")
            continue

        image_content_str = ', '.join(record['image_content'])
        input_text = template.format(
            image_content_str,
            record['question'],
            record['gt_answer'],
            record['model_answer']
        )

        response_content = None
        error_info = None
        max_retries = 3
        retry = 0

        while True:
            try:
                response = dashscope.Generation.call(
                    model=args.qwen_model,
                    prompt=input_text,
                    temperature=0.0,
                )
            except Exception as e:
                # 网络/未知异常，有限重试
                retry += 1
                print(f"\n网络或未知错误 (样本 {i}, 第 {retry} 次): {e}")
                if retry >= max_retries:
                    print(f"样本 {i} 网络异常重试超过 {max_retries} 次，跳过此样本。")
                    error_info = {
                        "error_code": "NetworkError",
                        "error_message": str(e),
                    }
                    break
                time.sleep(10)
                continue

            if response.status_code == HTTPStatus.OK:
                response_content = response.output.text
                break

            # 内容审查失败：不会因为重试而成功，直接跳过该样本
            if response.code == "DataInspectionFailed":
                print(f"\n样本 {i} 被内容安全审查拦截，跳过此样本。")
                print(f"ID: {response.request_id}, Msg: {response.message}")
                error_info = {
                    "error_code": response.code,
                    "error_message": response.message,
                }
                break

            # 其他错误：有限重试
            retry += 1
            print(
                f"\nAPI返回错误 (样本 {i}, 第 {retry} 次)！"
                f"ID: {response.request_id}, Code: {response.code}, Msg: {response.message}"
            )
            if retry >= max_retries:
                print(f"样本 {i} API 调用错误重试超过 {max_retries} 次，跳过此样本。")
                error_info = {
                    "error_code": response.code,
                    "error_message": response.message,
                }
                break
            time.sleep(10)

        # 统一记录结果：成功则有 qwen_response，失败则只记录错误信息
        record_out = {
            "sample_index": i,
        }
        if response_content is not None:
            record_out["qwen_response"] = response_content
        if error_info is not None:
            record_out.update(error_info)

        raw_responses.append(record_out)
        time.sleep(1)

    # 第六步：将新生成的结果写入文件
    try:
        with open(args.evaluation, 'w', encoding='utf-8') as f:
            json.dump(raw_responses, f, indent=2, ensure_ascii=False)
        print(f"\nQwen的详细评估回复已成功保存到: {args.evaluation}")
    except IOError as e:
        print(f"\n致命错误：无法写入评估文件 '{args.evaluation}'. 错误信息: {e}")
        return

    # 第七步：对新生成的结果进行分析
    analyze_results(raw_responses, args)


def analyze_results(raw_responses, args):
    """
    独立的分析函数，用于解析评分并打印结果。
    只统计真正有评分的样本；API失败 / 内容审查拦截 / 解析失败的样本全部跳过。
    """
    print("\n--- 开始分析评估结果 ---")
    scores = []
    valid_indices = []   # 记录哪些 sample_index 参与统计
    skipped_samples = [] # 记录被跳过的样本 index

    for resp_data in raw_responses:
        idx = resp_data.get('sample_index', 'N/A')

        # 1) API 阶段已经标记错误的样本，完全跳过
        if resp_data.get("error_code") is not None:
            print(
                f"样本 {idx} 在API调用阶段失败 (error_code={resp_data.get('error_code')}), "
                f"在统计中跳过。"
            )
            skipped_samples.append(idx)
            continue

        response_text = resp_data.get('qwen_response', '')
        if not response_text or not response_text.strip():
            print(f"样本 {idx} 没有有效的 qwen_response，在统计中跳过。")
            skipped_samples.append(idx)
            continue

        # 2) 正则解析评分
        match = re.search(r'#?\s*rating\s*:\s*\*+\s*(\d)', response_text, re.IGNORECASE)
        if not match:  # 如果第一个正则失败，尝试更宽松的
            match = re.search(r'rating\D*(\d)', response_text, re.IGNORECASE)

        if match:
            score = int(match.group(1))
            scores.append(score)
            valid_indices.append(idx)
        else:
            print(f"警告：在样本 {idx} 的回复中无法解析到评分，在统计中跳过该样本。")
            skipped_samples.append(idx)

    if not scores:
        print("\n错误：没有成功解析到任何评分，无法进行统计。")
        return

    # ========= 全局统计 =========
    hallucination = [1 if s < 3 else 0 for s in scores]

    print("\n--- 最终评估结果 ---")
    print(f"总样本数: {len(raw_responses)}")
    print(f"参与统计的有效样本数: {len(scores)}")
    print(f"被跳过的样本数: {len(skipped_samples)}")
    if skipped_samples:
        print(f"被跳过的样本 index 列表: {skipped_samples}")

    print('平均分 (Average score): {:.2f}'.format(sum(scores) / len(scores)))
    print('幻觉率 (Hallucination rate): {:.2f}%'.format(
        sum(hallucination) / len(hallucination) * 100
    ))

    # ========= 按类型分析分数 =========
    try:
        with open(args.response, 'r', encoding='utf-8') as f:
            records_for_type = json.load(f)

        # 原始每条记录的 question_type（与原始记录 index 对齐）
        question_types_list = [
            record.get('question_type', f'type_{i % 8}')
            for i, record in enumerate(records_for_type)
        ]
        unique_types = sorted(list(set(question_types_list)))
        scores_each_type = {q_type: [] for q_type in unique_types}

        # 使用 sample_index 对齐原始类型
        for sample_idx, score in zip(valid_indices, scores):
            if isinstance(sample_idx, int) and 0 <= sample_idx < len(question_types_list):
                q_type = question_types_list[sample_idx]
                scores_each_type[q_type].append(score)
            else:
                # 防御性提示
                print(f"警告：样本 index {sample_idx} 超出 question_types_list 范围，跳过类型统计。")

        print('各问题类型平均分:')
        for q_type in unique_types:
            type_scores = scores_each_type[q_type]
            if type_scores:
                avg_score = round(sum(type_scores) / len(type_scores), 2)
                print(f'  - {q_type}: {avg_score}')
            else:
                print(f'  - {q_type}: N/A (该类型全部样本被跳过或无评分)')
    except FileNotFoundError:
        print(f"警告：无法找到原始响应文件 '{args.response}'，无法按类型分析分数。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用通义千问模型评估LMM在MMHal-Bench上的表现。")
    parser.add_argument(
        '--response',
        type=str,
        default="/data/ruipeng.zhang/dpo_on/MMHal-Bench/output_7B/qwen2_5_vl_7B_p2dpo_2000.json",
        help='包含模型回答的JSON文件路径 (第一步的输出)'
    )
    parser.add_argument(
        '--evaluation',
        type=str,
        default="qwen_evaluation_details_7B/qwen2_5_vl_7B_p2dpo_2000_evaluation.json",
        help='保存Qwen评估原始回复的文件路径'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default="sk-e65a0bc89c634005b090b7324b96b9d6",
        help='你的通义千问API Key'
    )
    parser.add_argument(
        '--qwen-model',
        type=str,
        default='qwen-plus',
        help='用于评估的通义千问模型'
    )

    args = parser.parse_args()
    print(args.response)
    main(args)
