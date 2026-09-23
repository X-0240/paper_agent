# 并发压力测试：逐档提高同时请求数，看延迟与错误怎么变化
# 只读+调用接口；每档用简单问题，避免综述长任务把测试拖太久
import asyncio
import json
import time

import httpx

BASE="http://127.0.0.1:8000"
QUESTION="什么是自注意力机制？"


def token():
    return httpx.post(BASE+"/login",json={"username":"admin","password":"admin123"},timeout=20).json()["access_token"]


async def one(client,tok,i):
    #记录单个请求的首字节时间与总耗时
    t0=time.time()
    first=None
    try:
        async with client.stream("GET",BASE+"/ask/stream",
                                 params={"question":f"{QUESTION}（第{i}问）"},
                                 headers={"Authorization":"Bearer "+tok},
                                 timeout=300) as r:
            if r.status_code!=200:
                return {"code":r.status_code,"first":None,"total":time.time()-t0}
            async for line in r.aiter_lines():
                if line.startswith("data:") and first is None:
                    first=time.time()-t0
            return {"code":200,"first":first,"total":time.time()-t0}
    except Exception as e:
        return {"code":type(e).__name__,"first":first,"total":time.time()-t0}


async def run_level(n,tok):
    async with httpx.AsyncClient(timeout=300) as client:
        t0=time.time()
        results=await asyncio.gather(*[one(client,tok,i) for i in range(n)])
        wall=time.time()-t0
    ok=[r for r in results if r["code"]==200]
    codes={}
    for r in results:
        codes[str(r["code"])]=codes.get(str(r["code"]),0)+1
    totals=sorted(r["total"] for r in results)
    firsts=sorted(r["first"] for r in ok if r["first"] is not None)

    def pct(vals,p):
        return vals[min(len(vals)-1,int(len(vals)*p))] if vals else None

    return {
        "n":n,
        "wall":round(wall,1),
        "throughput":round(n/wall,2),
        "codes":codes,
        "first_p50":round(pct(firsts,0.5),1) if firsts else None,
        "total_p50":round(pct(totals,0.5),1),
        "total_p95":round(pct(totals,0.95),1),
    }


async def main():
    tok=token()
    print("="*84)
    print("并发压力测试：同问题、不同文案，逐档提高并发数")
    print("="*84)
    print("%-6s %-9s %-11s %-10s %-10s %-10s %s"%("并发","总墙钟","吞吐(题/秒)","首字节p50","总耗时p50","总耗时p95","状态码分布"))
    for n in [1,2,4,8,16,32]:
        r=await run_level(n,tok)
        print("%-6s %-9s %-11s %-10s %-10s %-10s %s"%(
            r["n"],r["wall"],r["throughput"],
            r["first_p50"] if r["first_p50"] is not None else "-",
            r["total_p50"],r["total_p95"],r["codes"]))
        await asyncio.sleep(2)


asyncio.run(main())
