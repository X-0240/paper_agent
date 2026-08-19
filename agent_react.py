import json
import logging
import sys
import time
from llm_api import safe_call_deepseek

#Windows控制台可能不支持论文中的数学符号，遇到无法编码字符时替换而不是崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if sys.stderr and hasattr(sys.stderr,"reconfigure"):
    sys.stderr.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#通用工具执行
def run_tool(name,args,tools):
    #异常兜底：工具不存在或执行失败都返回文本，不让Agent循环崩溃
    if name not in tools:
        return f"错误：工具{name}不存在，可用工具：{list(tools.keys())}"
    try:
        #调用前参数校验：Schema缺失字段或类型错误直接拒绝，防工具调用幻觉
        schema=tools[name].get("schema")
        if schema:
            err=_validate_args(schema,args)
            if err:
                return f"工具参数校验失败：{err}，请检查Action Input"
        result=tools[name]["func"](**args)
        if tools[name].get("is_worker"):
            answer,worker_history=result
            return {"answer":answer,"worker_history":worker_history}
        return result
    except Exception as e:
        return f"工具执行失败：{str(e)}"

def _validate_args(schema,args):
    #校验必填字段和类型，返回错误说明或None
    for field,rule in schema.items():
        value=args.get(field)
        if rule.get("required") and (value is None or value==""):
            return f"缺少必填参数：{field}"
        if value is not None and rule.get("type")=="str" and not isinstance(value,str):
            return f"参数类型错误：{field}应为字符串"
    return None

#通用Action解析
def parse_action(text):
    #多级解析：先试JSON，失败时按Finish正文兜底，兼容模型各种写法
    action=None; action_input={}; capture=False; answer_lines=[]
    for line in text.split("\n"):
        if line.startswith("Action:"):
            action=line.split(":",1)[1].strip()
            capture=True
        elif line.startswith("Action Input:"):
            raw=line.split(":",1)[1].strip()
            try:
                action_input=json.loads(raw)
            except:
                #Finish场景允许直接写正文，JSON解析失败时把原文当答案兜底
                action_input={"answer":raw} if action=="Finish" else {}
        elif capture and action=="Finish":
            stripped=line.strip()
            if not stripped or stripped.startswith("Thought:"):
                continue
            answer_lines.append(stripped)
    #模型可能漏写Action Input，直接把Action后的正文当答案
    if action=="Finish" and not action_input and answer_lines:
        action_input={"answer":"\n".join(answer_lines)}
    return action,action_input

#通用ReAct循环（每个Agent内部共用的思考-行动-观察内核）
def react(question,tools,system_prompt,max_steps=6,timeout=120,repeat_limit=3):
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":question}
    ]
    history=[]
    start=time.time(); action_trace=[]
    for step in range(max_steps):
        #时间预算：整体超时直接优雅退出，防止烧钱
        if time.time()-start>=timeout:
            logger.warning("任务超时，终止循环")
            return "任务超时",history
        logger.info(f"步骤{step+1}:调用LLM")
        #输出上限：Agent动作文本很短，封顶防异常长文烧钱
        result=safe_call_deepseek(messages,max_tokens=2048)
        ai_text=result["choices"][0]["message"]["content"] or ""
        if not ai_text.strip():
            logger.warning("LLM输出为空，要求重新输出")
            messages.append({"role":"user","content":"你的输出为空，请严格按格式重新输出"})
            continue
        logger.info(f"LLM输出:{ai_text}")

        action,action_input=parse_action(ai_text)
        #记录每一步的Action和输入，形成可观测日志
        history.append({"step":step+1,"action":action,"input":action_input})

        if action=="Finish":
            answer=action_input.get("answer","")
            #兜底：子Agent工具漏写答案时，用上一个工具观测结果
            if not answer and len(history)>=2:
                prev=history[-2]
                if prev.get("action") in tools and tools[prev["action"]].get("is_worker"):
                    answer=prev.get("observation","")
            return answer,history
        #重复动作检测：连续相同工具+参数直接终止，防死循环
        if action is not None:
            action_trace.append((action,json.dumps(action_input,sort_keys=True)))
            if len(action_trace)>=repeat_limit and len(set(action_trace[-repeat_limit:]))==1:
                logger.warning(f"连续{repeat_limit}次相同动作，终止循环")
                return f"连续{repeat_limit}次相同动作，终止任务",history
        if action is None:
            messages.append({"role":"user","content":"你的输出缺少Action，请只按以下格式重新输出，不要输出正文：\nThought: ...\nAction: Finish\nAction Input: {\"answer\": \"你的回答\"}"})
            continue
        if action not in tools:
            #未知工具：提示可用列表让模型重新选择，不中断整轮任务
            messages.append({"role": "user", "content": f"工具{action}不存在，可用工具：{list(tools.keys())}，请重新选择"})
            continue
        observation=run_tool(action,action_input,tools)
        #Observation超长时截断，防上下文膨胀；LLM摘要压缩留待增强
        if isinstance(observation,str) and len(observation)>1500:
            observation=observation[:1000]+f"\n...[Observation过长，已截断，原文{len(observation)}字]"
        #如果是子Agent工具，worker_history记进当前步骤，observation只用answer
        if isinstance(observation, dict) and "worker_history" in observation:
            history[-1]["worker_history"] = observation["worker_history"]
            observation = observation["answer"]
        logger.info(f"工具返回:{observation[:100]}")
        history[-1]["observation"]=observation

        messages.append({"role":"assistant","content":ai_text})
        #把工具结果作为Observation追加进上下文，供下一步思考使用
        messages.append({"role":"user","content":f"Observation: {observation}"})
        #历史裁剪：只保留系统+初始问题+最近3轮，多步循环不无限膨胀
        if len(messages)>8:
            messages=[messages[0],messages[1]]+messages[-6:]
    return "达到最大步骤数",history
