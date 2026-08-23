from agent_react import parse_action, react

def test_json_finish():
    #标准格式：Action+Action Input JSON
    text='Action: Finish\nAction Input: {"answer": "hello"}'
    action,action_input=parse_action(text)
    assert action=="Finish"
    assert action_input=={"answer":"hello"}

def test_finish_without_input_label():
    #模型漏写Action Input时，把正文当答案兜底
    text="Action: Finish\n答案是42"
    action,action_input=parse_action(text)
    assert action=="Finish"
    assert action_input.get("answer")=="答案是42"

def test_tool_bad_json():
    #非Finish场景JSON解析失败时不给答案字段，避免误当最终回答
    text="Action: Search\nAction Input: {bad json}"
    action,action_input=parse_action(text)
    assert action=="Search"
    assert action_input=={}

def test_validate_args_rejects_unknown():
    #多余参数必须被拒绝，防止LLM发明参数浪费步数
    from agent_react import run_tool
    tools={"Search":{"func":lambda q:"found","schema":{"q":{"type":"str","required":True}}}}
    out=run_tool("Search",{"q":"x","max_results":10},tools)
    assert "未知参数" in out

def test_react_on_step_callback(monkeypatch):
    #on_step回调在每次工具执行后收到完整history条目
    calls=[]
    def fake_llm(messages,**kw):
        if len(messages)<5:
            return {"choices":[{"message":{"content":"Action: Search\nAction Input: {\"q\":\"x\"}"}}]}
        return {"choices":[{"message":{"content":"Action: Finish\nAction Input: {\"answer\":\"done\"}"}}]}
    monkeypatch.setattr("agent_react.safe_call_deepseek",fake_llm)
    tools={"Search":{"func":lambda q:"found","description":"","schema":{"q":{"type":"str","required":True}}}}
    answer,history=react("q",tools,"system",max_steps=3,on_step=calls.append)
    assert answer=="done"
    assert calls
    assert calls[0]["action"]=="Search"
    assert "observation" in calls[0]
