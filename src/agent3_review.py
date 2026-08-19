import json
import logging
import os
import sys
from agent2_parse import CARD_DIR, extract_json
from llm_api import safe_call_deepseek

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

VERIFY_PROMPT="""你是事实一致性校验Agent。根据以下论文卡片，交叉校验事实冲突，只输出JSON，不要解释：
{
  "conflicts": [
    {
      "papers": ["论文A", "论文B"],
      "type": "指标冲突/结论矛盾/过时结论/信息不一致",
      "detail": "冲突内容说明",
      "severity": "severe 或 minor"
    }
  ]
}
规则：
1. 相同指标不同数值、互相矛盾的结论属于严重冲突（severe）
2. 观点分歧、详略不同属于轻微分歧（minor）
3. 没有冲突就输出 {"conflicts": []}"""

RETRIEVE_VERIFY_PROMPT="""你是事实冲突仲裁Agent。根据检索到的原文证据，判断冲突是否成立，只输出JSON，不要解释：
{
  "resolved": true 或 false,
  "verdict": "real_conflict 或 misunderstanding 或 card_error 或 insufficient_evidence",
  "reason": "一句话说明"
}
规则：
- real_conflict：原文确实存在同一指标不同数值或矛盾结论
- misunderstanding：看似冲突，实际是不同设置/不同口径（如单模型 vs 集成模型）
- card_error：某张卡片提取错误，原文支持另一方
- insufficient_evidence：证据不足，无法裁决"""

REVIEW_PROMPT="""你是科研综述生成Agent。根据论文卡片和事实校验结果，生成完整科研综述（Markdown）：
1. 研究背景概述
2. 各论文要点
3. 横向对比与技术演进（如果提供了对比结果，直接整合）
4. 事实冲突说明：严重冲突如实标注"未解决"，轻微分歧备注即可
5. 结论与局限
要求：引用论文名和章节来源，不编造卡片之外的内容。只输出综述正文。"""

def load_cards(paper_names):
    cards=[]
    for name in paper_names:
        path=os.path.join(CARD_DIR,f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少卡片：{name}，请先运行agent2_parse生成")
        cards.append(json.load(open(path,encoding="utf-8")))
    return cards

def call_verify(messages,max_attempts=3):
    #结构化校验输出：解析失败或缺conflicts字段就纠正重试
    for attempt in range(max_attempts):
        result=safe_call_deepseek(messages,temperature=0.2,max_tokens=2500)
        content=result["choices"][0]["message"]["content"] or ""
        try:
            parsed=extract_json(content)
        except ValueError:
            if attempt==max_attempts-1:
                raise
            messages.append({"role":"user","content":"你上次的输出不是合法JSON，请只输出JSON对象，不要解释。"})
            continue
        if not isinstance(parsed,dict) or "conflicts" not in parsed:
            if attempt==max_attempts-1:
                raise ValueError("校验结果缺少conflicts字段")
            messages.append({"role":"user","content":"输出必须包含conflicts字段，请重新只输出JSON。"})
            continue
        return parsed
    raise ValueError("校验结果生成失败")

def deduplicate_conflicts(conflicts):
    #多轮回环可能重复上报同一冲突，LLM会改类型和措辞，按"论文对"合并最稳
    seen=set(); unique=[]
    for c in conflicts:
        papers=tuple(sorted(c.get("papers",[])))
        if papers not in seen:
            seen.add(papers)
            unique.append(c)
    return unique

def resolve_conflict_with_retrieval(conflict):
    #二次检索验证：冲突不能只看卡片，要回原文取证
    from rag_tool import search_papers_structured
    query=conflict.get("detail","")[:200] or " ".join(conflict.get("papers",[]))
    results=search_papers_structured(query,k=5)
    if not results:
        return {"resolved":False,"verdict":"insufficient_evidence","reason":"无检索结果"}
    evidence="\n\n".join(
        f"[{r['source']} - {r['section']}]\n{r['text']}"
        for r in results[:5]
    )
    messages=[
        {"role":"system","content":RETRIEVE_VERIFY_PROMPT},
        {"role":"user","content":f"冲突：{json.dumps(conflict,ensure_ascii=False)}\n\n检索到的原文证据：\n{evidence}"}
    ]
    for attempt in range(3):
        result=safe_call_deepseek(messages,temperature=0.1,max_tokens=1500)
        content=result["choices"][0]["message"]["content"] or ""
        try:
            data=extract_json(content)
            if isinstance(data,dict) and "verdict" in data:
                return data
        except Exception:
            pass
        messages.append({"role":"user","content":"你上次的输出不是合法JSON，请只输出JSON对象。"})
    return {"resolved":False,"verdict":"insufficient_evidence","reason":"仲裁输出解析失败"}

def verify_with_retrieval(cards,max_loops=3):
    #校验+二次检索：严重冲突回原文取证，消解/确认/降级
    conflicts=verify_cards(cards,max_loops)
    final=[]
    for c in conflicts:
        if c.get("severity")!="severe":
            final.append(c)
            continue
        verdict=resolve_conflict_with_retrieval(c)
        if verdict.get("verdict") in ("misunderstanding","card_error"):
            c["severity"]="minor"
            c["detail"]=c.get("detail","")+f" [二次检索：{verdict.get('reason','')}]"
        elif verdict.get("verdict")=="real_conflict":
            c["detail"]=c.get("detail","")+f" [二次检索确认：{verdict.get('reason','')}]"
        else:
            c["detail"]=c.get("detail","")+f" [二次检索证据不足：{verdict.get('reason','')}]"
        final.append(c)
    return final

def verify_cards(cards,max_loops=3):
    #最多3轮回环：严重冲突时只复核冲突论文，防同一批劣质素材反复处理
    conflicts_all=[]; current=cards
    for loop in range(max_loops):
        cards_text=json.dumps(current,ensure_ascii=False,indent=2)[:12000]
        if loop==0:
            user_text=cards_text
        else:
            user_text=f"以下卡片上一轮存在严重冲突，请重点复核指标与结论，无法判定就保留冲突：\n{cards_text}"
        messages=[
            {"role":"system","content":VERIFY_PROMPT},
            {"role":"user","content":user_text}
        ]
        data=call_verify(messages)
        conflicts=data.get("conflicts",[])
        conflicts_all.extend(conflicts)
        severe=[c for c in conflicts if c.get("severity")=="severe"]
        if not severe:
            break
        #只保留冲突论文进入下一轮复核
        bad=[]
        for c in severe:
            for title in c.get("papers",[]):
                if title not in bad:
                    bad.append(title)
        current=[c for c in current if c.get("title") in bad]
        if not current:
            break
        if loop<max_loops-1:
            logger.info(f"第{loop+1}轮发现{len(severe)}条严重冲突，进入下一轮复核：{bad}")
    return deduplicate_conflicts(conflicts_all)

def mark_card_risks(cards,conflicts):
    #按冲突结果给卡片打风险标记，供综述引用
    for card in cards:
        card["risk"]="无"
    for c in conflicts:
        if c.get("severity")!="severe":
            continue
        for title in c.get("papers",[]):
            for card in cards:
                if card.get("title")==title:
                    card["risk"]=c.get("type","风险")

def generate_review(cards,conflicts,comparison=None):
    mark_card_risks(cards,conflicts)
    data={"cards":cards,"conflicts":conflicts}
    if comparison:
        data["comparison"]=comparison
    payload=json.dumps(data,ensure_ascii=False,indent=2)[:14000]
    messages=[
        {"role":"system","content":REVIEW_PROMPT},
        {"role":"user","content":payload}
    ]
    result=safe_call_deepseek(messages,temperature=0.3,max_tokens=4000)
    return result["choices"][0]["message"]["content"] or ""

def run_review(paper_names,max_loops=3,comparison=None):
    cards=load_cards(paper_names)
    conflicts=verify_cards(cards,max_loops)
    review=generate_review(cards,conflicts,comparison)
    return review,conflicts

def _human_confirm(conflicts):
    #严重冲突未解决时人工确认：继续/剔除论文/中止
    severe=[c for c in conflicts if c.get("severity")=="severe"]
    if not severe:
        return None
    print("\n检测到严重事实冲突，需要人工确认：")
    for c in severe:
        print(f"- {c.get('type')}：{c.get('papers')} {c.get('detail','')}")
    print("1=继续生成（综述标注未解决） 2=剔除某篇论文重新生成 3=中止")
    choice=input("输入1/2/3：").strip()
    if choice=="3":
        return "abort"
    if choice=="2":
        return input("输入要剔除的论文名：").strip()
    return None

def agent_3(papers,comparison,human_confirm=False):
    #Agent-3：事实校验+综述生成；human_confirm开启时严重冲突先问用户
    cards=load_cards(papers)
    conflicts=verify_with_retrieval(cards)
    if human_confirm:
        decision=_human_confirm(conflicts)
        if decision=="abort":
            return {"review":"已中止：存在严重事实冲突，等待人工确认","conflicts":conflicts}
        if decision:
            #剔除用户指定的论文后重新校验和生成
            papers=[p for p in papers if p!=decision]
            cards=load_cards(papers)
            conflicts=verify_with_retrieval(cards)
    review=generate_review(cards,conflicts,comparison)
    return {"review":review,"conflicts":conflicts}

if __name__=="__main__":
    if len(sys.argv)<2:
        print("用法：python agent3_review.py 论文名1 论文名2 ...")
        sys.exit(0)
    names=[a.replace(".pdf","") for a in sys.argv[1:]]
    review,conflicts=run_review(names)
    severe=[c for c in conflicts if c.get("severity")=="severe"]
    print(f"校验发现{len(conflicts)}条冲突，其中严重{len(severe)}条\n")
    for c in conflicts:
        print(f"[{c.get('severity')}] {c.get('type')}：{c.get('papers')} {c.get('detail','')}")
    print("\n===== 综述 =====")
    print(review)
