from collections import OrderedDict

def should_rerank(items,k=5,min_margin=0.1):
    #粗排top1与top-k分差够大时跳过重排，控制延迟
    if len(items)<=1:
        return False
    k=min(k,len(items))
    top1=float(items[0][1])
    topk=float(items[k-1][1])
    return (top1-topk)<min_margin

def cache_key(query,snapshot=""):
    #快照+query作为键，索引更新后旧缓存自动失效
    return f"{snapshot}||{query.strip().lower()}"

class RerankCache:
    #进程内LRU：避免同一query重复跑Cross-Encoder
    def __init__(self,max_size=256):
        self.max_size=max_size
        self.data=OrderedDict()

    def get(self,key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def set(self,key,value):
        self.data[key]=value
        self.data.move_to_end(key)
        while len(self.data)>self.max_size:
            self.data.popitem(last=False)

    def clear(self):
        self.data.clear()
