import datetime
import json
from llm_api import USAGE_FILE

def main():
    #成本报表：按日汇总并打印当日明细条数
    today=datetime.date.today().isoformat()
    daily={}
    try:
        with open(USAGE_FILE,encoding="utf-8") as f:
            for line in f:
                try:
                    item=json.loads(line)
                    day=item.get("date","")
                    daily[day]=daily.get(day,{"calls":0,"tokens":0,"cost":0.0})
                    daily[day]["calls"]+=1
                    daily[day]["tokens"]+=item.get("total_tokens",0)
                    daily[day]["cost"]+=item.get("cost",0)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("暂无用量记录")
        return
    for day in sorted(daily,reverse=True):
        d=daily[day]
        mark=" (今日)" if day==today else ""
        print(f"{day}{mark} 调用{d['calls']}次 token={d['tokens']} 约{d['cost']:.4f}元")

if __name__=="__main__":
    main()
