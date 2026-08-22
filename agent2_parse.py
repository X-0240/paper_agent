import json
import logging
import os
from llm_api import safe_call_deepseek

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

CARD_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"cards")
os.makedirs(CARD_DIR,exist_ok=True)

#以下两函数复用paper_rag的章节提取逻辑（PyMuPDF blocks按标题切分）
def get_paper_sections(paper_name):
    #旧接口适配：章节读取统一收敛到doc_ingest
    from doc_ingest import load_sections
    sections=load_sections(paper_name)
    if not sections:
        raise FileNotFoundError(f"找不到论文章节：{paper_name}")
    return [(s.get("title",""),s.get("text","")) for s in sections]

def get_paper_title(paper_name):
    #旧接口适配：标题映射统一收敛到doc_ingest
    from doc_ingest import paper_title
    return paper_title(paper_name)

CARD_PROMPT="""你是论文结构化解析Agent。根据给定论文章节内容，提取结构化信息，只输出JSON，不要输出其他内容：
{
  "title": "论文标题",
  "background": "研究背景和要解决的问题",
  "method": "核心方法或模型结构",
  "innovation": ["创新点1", "创新点2"],
  "experiments": "实验设置、数据集和主要指标",
  "conclusion": "主要结论",
  "usage":"适用场景",
  "limitations": "作者承认的局限或论文不足"
}
如果某字段原文没有，写"未提及"，不要编造。"""

def select_card_input(sections,max_chars=8000):
    #上下文裁剪：优先摘要/引言/方法/结论，单节截断，控制LLM输入
    priority=[]
    for title,text in sections:
        low=title.lower()
        if any(k in low for k in ["abstract","introduction","conclusion","method","architecture","approach","related work"]):
            priority.append((title,text))
    if not priority:
        priority=sections[:5]
    parts=[]; total=0
    for title,text in priority:
        part=text[:2000]
        if total+len(part)>max_chars:
            part=part[:max_chars-total]
        parts.append(f"【{title}】\n{part}")
        total+=len(part)
        if total>=max_chars:
            break
    return "\n".join(parts)

def extract_json(text):
    #LLM可能包代码块或加说明，只取第一个{到最后一个}之间的JSON
    text=text.strip()
    if text.startswith("```"):
        text=text.strip("`")
        if text.startswith("json"):
            text=text[4:]
    start=text.find("{")
    end=text.rfind("}")
    if start==-1 or end==-1:
        raise ValueError("LLM未输出JSON")
    return json.loads(text[start:end+1])

REQUIRED_FIELDS={"title","background","method","innovation","experiments","conclusion","limitations"}

def build_paper_card(paper_name):
    #卡片缓存：同一论文不重复调用LLM
    card_path=os.path.join(CARD_DIR,f"{paper_name}.json")
    if os.path.exists(card_path):
        return json.load(open(card_path,encoding="utf-8"))
    sections=get_paper_sections(paper_name)
    input_text=select_card_input(sections)
    messages=[
        {"role":"system","content":CARD_PROMPT},
        {"role":"user","content":f"论文：{get_paper_title(paper_name) or paper_name}\n以下是论文章节内容：\n{input_text}"}
    ]
    card=None
    #LLM输出不稳定：JSON解析失败或字段缺失就纠正重试，最多3次
    for attempt in range(3):
        result=safe_call_deepseek(messages,temperature=0.2,max_tokens=2000)
        content=result["choices"][0]["message"]["content"] or ""
        try:
            parsed=extract_json(content)
        except ValueError:
            if attempt==2:
                raise
            messages.append({"role":"user","content":"你上次的输出不是合法JSON，请只输出一个JSON对象，不要任何解释或代码块。"})
            continue
        #空卡片防御：解析成功但缺字段的卡片不能入库
        missing=REQUIRED_FIELDS if not isinstance(parsed,dict) else REQUIRED_FIELDS-set(parsed.keys())
        if missing:
            if attempt==2:
                raise ValueError(f"卡片缺少必要字段：{sorted(missing)}")
            messages.append({"role":"user","content":f"你输出的JSON缺少字段：{sorted(missing)}，请按模板输出完整字段，只输出JSON。"})
            continue
        card=parsed
        break
    if card is None:
        raise ValueError("卡片生成失败")
    with open(card_path,"w",encoding="utf-8") as f:
        json.dump(card,f,ensure_ascii=False,indent=2)
    logger.info(f"已生成卡片:{paper_name}")
    return card

COMPARE_PROMPT="""你是多论文横向对比Agent。根据以下论文结构化卡片，生成横向对比报告：
1. Markdown表格，列为：研究问题、核心方法、创新点、实验数据集与指标、适用场景、局限
2. 单独给出技术演进脉络（这些论文之间的继承与改进关系）
只输出报告正文，不要输出额外解释。"""

def compare_papers(paper_names):
    #多篇卡片合并对比：输入是同构JSON卡片，输出Markdown对比矩阵
    cards=[build_paper_card(name) for name in paper_names]
    cards_text=json.dumps(cards,ensure_ascii=False,indent=2)
    #对比输入可能很大，截断保护
    cards_text=cards_text[:12000]
    messages=[
        {"role":"system","content":COMPARE_PROMPT},
        {"role":"user","content":cards_text}
    ]
    result=safe_call_deepseek(messages,temperature=0.2,max_tokens=3000)
    return result["choices"][0]["message"]["content"]

def agent_2(papers):
    #Agent-2：解析-对比（确定性流程，解析完才能对比）
    for name in papers:
        card_path=os.path.join(CARD_DIR,f"{name}.json")
        if not os.path.exists(card_path):
            logger.info(f"Agent-2 生成卡片：{name}")
            build_paper_card(name)
    comparison=compare_papers(papers)
    return {"papers":papers,"comparison":comparison}

if __name__=="__main__":
    #命令行入口：传论文名即可生成卡片并对比
    import sys
    if len(sys.argv)<2:
        print("用法：python agent2_parse.py 论文名1 论文名2 ...")
        print("示例：python agent2_parse.py Attention_Is_All_You_Need RoFormer_RoPE")
        sys.exit(0)
    names=[a.replace(".pdf","") for a in sys.argv[1:]]
    report=compare_papers(names)
    print(report)
