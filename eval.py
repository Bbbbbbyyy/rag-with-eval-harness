# eval.py
import time
from rag import ChatPDF

TEST_CASES = [
    # ── 关键词直接能过的题 ──
    {
        "question": "这个项目部署在哪个城市？",
        "expected_contains": ["寿光", "山东"]
    },
    {
        "question": "系统用什么无线通信技术？",
        "expected_contains": ["LoRa"]
    },
    {
        "question": "土壤湿度低于多少会预警？",
        "expected_contains": ["30"]
    },
    {
        "question": "数据准确率是多少？",
        "expected_contains": ["98.5"]
    },
    {
        "question": "系统覆盖多少亩？",
        "expected_contains": ["200"]
    },
    {
        "question": "灌溉用水量减少了多少？",
        "expected_contains": ["25"]
    },

    # ── 故意让关键词失败、靠 LLM 裁判判断的题 ──
    {
        "question": "这个系统的预警方式是什么？",
        # 文档里写的是"短信和微信通知"，但模型很可能只答"微信"或"通知"
        "expected_contains": ["短消息"]
    },
    {
        "question": "系统的数据采集频率是多少？",
        # 文档写"每10分钟一次"，模型可能答"每十分钟"或"10min"
        "expected_contains": ["10", "分钟"]
    },
    {
        "question": "项目下一步的计划是什么？",
        # 文档写"AI病虫害识别"，模型可能答"人工智能识别病虫害"
        "expected_contains": ["AI", "病虫害", "识别"]
    },
    {
        "question": "传感器节点的数量是多少？",
        # 文档写"50个"，模型可能答"五十个"
        "expected_contains": ["五十"]
    },
    {
        "question": "系统用的是什么数据库？",
        # 文档写"MySQL"，模型可能答"关系型数据库 MySQL"
        "expected_contains": ["关系型数据库"]
    },
    {
        "question": "运维团队有几个人？",
        # 文档写"3人"，模型可能答"三人"
        "expected_contains": ["三人"]
    },
]


def llm_judge(pdf_assistant, question, answer, expected_keywords):
    """
    当关键词检查失败时，调用 LLM 做二次判断。
    返回 True 表示 LLM 认为回答准确，False 表示不准确。
    """
    judge_prompt = f"""
    你是一个评测裁判。请判断下面这个回答是否准确回答了问题。

    问题：{question}
    模型回答：{answer}
    预期关键词：{expected_keywords}

    判断标准：
    - 如果回答包含了预期关键词的意思，或者虽然没有原词但语义正确，算通过。
    - 如果回答完全答非所问，或者明显错误，算不通过。

    只回答一个字："是" 或 "否"，不要解释。
    """
    result, _ = pdf_assistant.ask(judge_prompt)
    return "是" in result


def evaluate():
    pdf_assistant = ChatPDF()
    pdf_assistant.ingest("data/pdfs/smart_agriculture.pdf")

    passed_count = 0
    keyword_passed = 0
    judge_passed = 0

    print(f"开始评测，共 {len(TEST_CASES)} 个测试用例...\n")

    for i, case in enumerate(TEST_CASES, 1):
        start_time = time.time()
        try:
            answer, _ = pdf_assistant.ask(case["question"])
            duration = time.time() - start_time

            # 第一级：关键词检查
            check_passed = True
            for keyword in case["expected_contains"]:
                if keyword not in answer:
                    check_passed = False
                    break

            if check_passed:
                keyword_passed += 1
                passed_count += 1
                print(f"{i}/{len(TEST_CASES)} [关键词通过] {case['question'][:20]}... ({duration:.2f}s)")
            else:
                # 第二级：LLM 裁判
                judge_result = llm_judge(
                    pdf_assistant,
                    case["question"],
                    answer,
                    case["expected_contains"]
                )
                if judge_result:
                    judge_passed += 1
                    passed_count += 1
                    print(f"{i}/{len(TEST_CASES)} [LLM裁判通过] {case['question'][:20]}...")
                else:
                    print(f"{i}/{len(TEST_CASES)} [失败] {case['question'][:20]}...")
                    print(f"    回答: {answer[:100]}...")

        except Exception as e:
            print(f"{i}/{len(TEST_CASES)} [错误] {e}")

    pass_rate = (passed_count / len(TEST_CASES)) * 100
    print(f"\n{'='*50}")
    print(f"评测完成！通过率: {pass_rate:.1f}% ({passed_count}/{len(TEST_CASES)})")
    print(f"  其中关键词通过: {keyword_passed}，LLM裁判通过: {judge_passed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    evaluate()
