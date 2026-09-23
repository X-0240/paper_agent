# 验证准入：并发过载时是"快速明确拒绝"还是"全部挂着等超时"
import asyncio
import json
import time

import httpx

B="http://127.0.0.1:8000"


def tok():
    return httpx.post(B+"/login",json={"username":"admin","password":"admin123"},timeout=20).json()["access_token"]


async def one(client,t,i):
    t0=time.time()
    first=None
    try:
        async with client.stream("GET",B+"/ask/stream",
                                 params={"question":"什么是自注意力机制？第%d问"%i},
                                 headers={"Authorization":"Bearer "+t},
                                 timeout=300) as r:
            code=r.status_code
            if code!=200:
                body=await r.aread()
                return {"code":code,"first":time.time()-t0,"total":time.time()-t0,
                        "detail":body.decode("utf-8","ignore")[:70]}
            async for line in r.aiter_lines():
                if line.startswith("data:") and first is None:
                    first=time.time()-t0
            return {"code":200,"first":first,"total":time.time()-t0,"detail":""}
    except Exception as e:
        return {"code":type(e).__name__,"first":first,"total":time.time()-t0,"detail":str(e)[:60]}


async def level(n,t):
    async with httpx.AsyncClient(timeout=300) as c:
        t0=time.time()
        res=await asyncio.gather(*[one(c,t,i) for i in range(n)])
        wall=time.time()-t0
    codes={}
    for r in res:
        k=str(r["code"])
        codes[k]=codes.get(k,0)+1
    rejected=[r for r in res if r["code"]!=200]
    ok=[r for r in res if r["code"]==200]
    return {
        "n":n,"wall":round(wall,1),"codes":codes,
        "reject_max_wait":round(max([r["total"] for r in rejected]),2) if rejected else None,
        "ok_p50":round(sorted(r["total"] for r in ok)[len(ok)//2],1) if ok else None,
        "sample_detail":next((r["detail"] for r in rejected if r["detail"]),""),
    }


async def main():
    t=tok()
    print("="*84)
    print("并发过载下的准入表现（max_inflight=8）")
    print("="*84)
    print("%-6s %-9s %-26s %-14s %-12s %s"%("并发","总墙钟","状态码分布","拒绝最长等待","成功p50","拒绝原因样例"))
    for n in [6,12,20]:
        r=await level(n,t)
        print("%-6s %-9s %-26s %-14s %-12s %s"%(
            r["n"],r["wall"],str(r["codes"]),
            r["reject_max_wait"] if r["reject_max_wait"] is not None else "-",
            r["ok_p50"] if r["ok_p50"] is not None else "-",
            r["sample_detail"]))
        await asyncio.sleep(3)
    print()
    print("说明：拒绝应该在 1 秒内返回；若「拒绝最长等待」很大，说明用户在干等而不是被明确拒绝")


asyncio.run(main())
