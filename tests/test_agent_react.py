from agent_react import parse_action

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
