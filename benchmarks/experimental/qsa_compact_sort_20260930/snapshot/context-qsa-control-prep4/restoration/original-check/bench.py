#!/usr/bin/env python3
"""Fixed-token SSE workload; no EOS truncation, unique cache salts, raw metrics."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,statistics,subprocess,time,uuid
import requests
BASE='http://127.0.0.1:8200'
ROOT=pathlib.Path(__file__).resolve().parent
TEXT='''A software team is implementing a geographic routing service. Roads form a directed graph. Each road segment has a length, speed limit, turn restriction and access class. The implementation must preserve deterministic tie breaking, check invalid inputs, and separate loading data from query execution. A route query returns the shortest valid travel-time path. Testing should include disconnected vertices, zero-length edges, repeated queries, and changes to road permissions. Explain the reasoning clearly and include executable Python with comments.\n'''

def token_ids(n,idx):
    # Different leading tokens prevent cross-request prefix reuse; source text fixed across arms.
    s=f'Workload example {idx:06d}.\n'+TEXT*(n//50+20)
    r=requests.post(BASE+'/tokenize',json={'model':'flash-next','prompt':s},timeout=30); r.raise_for_status()
    ids=r.json()['tokens'][:n]
    if len(ids)!=n: raise ValueError('insufficient tokens')
    return ids

def metrics():
    r=requests.get(BASE+'/metrics',timeout=10);r.raise_for_status();return r.text

def req(ids,out,idx,mode):
    data={'model':'flash-next','prompt':ids,'max_tokens':out,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},'cache_salt':uuid.uuid4().hex,'seed':1729+idx,'temperature':0 if mode=='greedy' else 0.6,'top_p':1 if mode=='greedy' else 0.95,'top_k':-1 if mode=='greedy' else 20,'return_token_ids':True}
    start=time.perf_counter();wall=time.time();chunks=[];text=[];returned_ids=[];usage=None;finish=None
    try:
        with requests.post(BASE+'/v1/completions',json=data,stream=True,timeout=(10,240)) as r:
            r.raise_for_status()
            transfer_encoding=r.headers.get('Transfer-Encoding','').lower()
            if 'chunked' not in transfer_encoding: raise RuntimeError('SSE benchmark expects chunked HTTP transport')
            for line in r.iter_lines(chunk_size=4096):
                if not line.startswith(b'data:'):continue
                v=line[5:].strip()
                if v==b'[DONE]':break
                obj=json.loads(v)
                if 'error' in obj:raise RuntimeError(str(obj['error']))
                if obj.get('usage'):usage=obj['usage']
                for c in obj.get('choices',[]):
                    tids=c.get('token_ids') or []
                    if c.get('text') or tids: chunks.append({'t':time.perf_counter()-start,'tokens':len(tids)});text.append(c.get('text',''));returned_ids.extend(tids)
                    if c.get('finish_reason'):finish=c['finish_reason']
        elapsed=time.perf_counter()-start
        if not usage or usage['completion_tokens']!=out or usage['prompt_tokens']!=len(ids):raise RuntimeError(f'usage mismatch {usage}')
        first=chunks[0]['t'] if chunks else None
        return {'client_version':2,'sse_chunk_bytes':4096,'http_transfer_encoding':transfer_encoding,'ok':True,'idx':idx,'start_wall':wall,'elapsed':elapsed,'ttft':first,'tpot_conventional':(elapsed-first)/(out-1) if first is not None and out>1 else None,'usage':usage,'finish':finish,'chunks':chunks,'output_sha256':hashlib.sha256(''.join(text).encode()).hexdigest(),'output_token_ids':returned_ids}
    except Exception as e:return {'ok':False,'idx':idx,'start_wall':wall,'elapsed':time.perf_counter()-start,'error':str(e)}

def run_case(label,n,out,c,r,mode):
    prompts=[token_ids(n,i) for i in range(r)]
    before=metrics();t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool: rows=list(pool.map(lambda i:req(prompts[i],out,i,mode),range(r)))
    elapsed=time.perf_counter()-t;after=metrics();good=[x for x in rows if x['ok']]
    p=ROOT/(label+'.json')
    summ={'client_version':2,'label':label,'input_tokens':n,'output_tokens':out,'concurrency':c,'requests':r,'success':len(good),'wall_seconds':elapsed,'output_tps':sum(x['usage']['completion_tokens'] for x in good)/elapsed,'input_tps':sum(x['usage']['prompt_tokens'] for x in good)/elapsed,'ttft_median':statistics.median(x['ttft'] for x in good) if good else None,'tpot_median':statistics.median(x['tpot_conventional'] for x in good) if out>1 and good else None}
    p.write_text(json.dumps({'summary':summ,'mode':mode,'rows':rows},indent=2))
    (ROOT/(label+'.metrics_before.txt')).write_text(before);(ROOT/(label+'.metrics_after.txt')).write_text(after)
    print(json.dumps(summ),flush=True)
    if len(good)!=r:raise RuntimeError('failed requests')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--suite',choices=['short','latency','prefill','long','smoke'],default='short');a.add_argument('--mode',choices=['sample','greedy'],default='sample');o=a.parse_args()
    cases={'short':[(512,128,1,1),(512,256,1,3),(512,256,2,6),(512,256,4,8),(512,256,8,16)],'latency':[(512,128,1,1),(512,256,1,5)],'prefill':[(2048,1,1,1),(8192,1,1,3),(32768,1,1,3)],'long':[(32768,256,1,2),(65536,1,1,2)],'smoke':[(128,16,1,1)]}[o.suite]
    tele=open(ROOT/(o.tag+'.gpu.csv'),'w')
    p=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,nounits','-lms','500'],stdout=tele,stderr=subprocess.DEVNULL)
    try:
        for j,(n,out,c,r) in enumerate(cases):run_case(f'{o.tag}_{j}_p{n}_o{out}_c{c}',n,out,c,r,o.mode)
    finally:
        p.terminate();p.wait(timeout=5);tele.close()
