"""STEP-22 client: copy of step09-gdn-replay/client.py with request-id prefix step41a-."""
from __future__ import annotations
import argparse,datetime,hashlib,json,pathlib,statistics,subprocess,time,uuid
import requests
D=pathlib.Path(__file__).resolve().parent
BASE='http://127.0.0.1:8200'
KEYS=('vllm:prompt_tokens_total','vllm:generation_tokens_total','vllm:prefix_cache_hits_total','vllm:request_generation_tokens_count','vllm:spec_decode_num_drafts_total','vllm:spec_decode_num_draft_tokens_total','vllm:spec_decode_num_accepted_tokens_total')
def metrics():
 r=requests.get(BASE+'/metrics',timeout=10);r.raise_for_status();return r.text

def counters(text):
 out={k:0.0 for k in KEYS}
 for line in text.splitlines():
  if line and not line.startswith('#'):
   key=line.split('{',1)[0].split(' ',1)[0]
   if key in out:out[key]+=float(line.rsplit(' ',1)[1])
 return out

def prompt():return json.loads((D/'input-8192.json').read_text())['tokens'][:512]

def run(label,ids,out,allowed=None,logprobs=None,bias_token=None):
 path=D/(label+'.json')
 if path.exists():raise RuntimeError('Refusing overwrite: '+str(path))
 body={'request_id':'step41a-'+label+'-'+uuid.uuid4().hex,'model':'flash-next','prompt':ids,'max_tokens':out,'temperature':0,'top_p':1,'top_k':-1,'ignore_eos':True,'seed':1729,'cache_salt':uuid.uuid4().hex,'return_token_ids':True,'stream':True,'stream_options':{'include_usage':True}}
 if allowed is not None:body['allowed_token_ids']=allowed
 if bias_token is not None:body['logit_bias']={str(bias_token):100}
 if logprobs is not None:body.update(logprobs=logprobs,return_tokens_as_token_ids=True)
 path.with_suffix('.request.json').write_text(json.dumps(body,indent=2))
 before=metrics();b=counters(before);path.with_suffix('.metrics_before.txt').write_text(before)
 start=time.perf_counter();wall=time.time();chunks=[];token_ids=[];logs=[];text=[];usage=None;error=None;finish=None
 try:
  with requests.post(BASE+'/v1/completions',json=body,stream=True,timeout=(10,180)) as r:
   if r.status_code!=200:raise RuntimeError(f'HTTP {r.status_code}: {r.text[:3000]}')
   for line in r.iter_lines(chunk_size=4096):
    if not line.startswith(b'data:'):continue
    raw=line[5:].strip()
    if raw==b'[DONE]':break
    j=json.loads(raw)
    if 'error' in j:raise RuntimeError(str(j['error']))
    if j.get('usage'):usage=j['usage']
    for choice in j.get('choices',[]):
     ts=choice.get('token_ids') or []
     if ts or choice.get('text'):chunks.append({'t':time.perf_counter()-start,'n':len(ts)});token_ids.extend(ts);text.append(choice.get('text',''))
     if choice.get('logprobs') is not None:logs.append(choice['logprobs'])
     if choice.get('finish_reason'):finish=choice['finish_reason']
 except Exception as e:error=str(e)
 elapsed=time.perf_counter()-start;after=metrics();a=counters(after);delta={k:a[k]-b[k] for k in KEYS};path.with_suffix('.metrics_after.txt').write_text(after)
 checks={'http_and_stream_ok':error is None,'usage_ok':usage is not None and usage.get('prompt_tokens')==len(ids) and usage.get('completion_tokens')==out,'returned_tokens_ok':len(token_ids)==out,'input_counter_ok':delta[KEYS[0]]==len(ids),'output_counter_ok':delta[KEYS[1]]==out,'request_counter_ok':delta[KEYS[3]]==1,'no_prefix_hits':delta[KEYS[2]]==0,'allowed_respected':allowed is None or all(t in allowed for t in token_ids),'forced_bias_respected':bias_token is None or token_ids==[bias_token]*out}
 ttft=chunks[0]['t'] if chunks else None
 result={'label':label,'utc':datetime.datetime.fromtimestamp(wall,datetime.timezone.utc).isoformat(),'start_wall':wall,'prompt_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'input_tokens':len(ids),'output_tokens':out,'allowed_token_ids':allowed,'bias_token':bias_token,'logprobs_requested':logprobs,'elapsed_s':elapsed,'ttft_s':ttft,'tpot_ms':1000*(elapsed-ttft)/(out-1) if ttft is not None and out>1 else None,'output_tps':len(token_ids)/elapsed,'output_token_ids':token_ids,'chunks':chunks,'text':''.join(text),'logprobs':logs,'usage':usage,'finish':finish,'metrics_delta':delta,'checks':checks,'error':error}
 path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({k:result[k] for k in ('label','elapsed_s','ttft_s','output_tps','metrics_delta','checks','error')},ensure_ascii=False),flush=True)
 with (D/'PROGRESS.md').open('a') as f:f.write(f"\n`{label}` 完成，checks={all(checks.values())}, TTFT={ttft}，E2E={elapsed}。\n")
 if not all(checks.values()):raise RuntimeError('Request validation failed: '+label)
 return result

def probe():
 rows=[]
 for tid in (1000,50000,100000,200000):
  r=run('probe-'+str(tid),prompt(),64,[tid]);rows.append({'token':tid,'accepted':r['metrics_delta'][KEYS[6]],'drafts':r['metrics_delta'][KEYS[4]],'draft_tokens':r['metrics_delta'][KEYS[5]]})
  if rows[-1]['accepted']==0:
   r2=run('probe-full-'+str(tid),prompt(),128,[tid]);rows[-1]['full128_accepted']=r2['metrics_delta'][KEYS[6]]
   if rows[-1]['full128_accepted']==0:
    (D/'fixed-selection.json').write_text(json.dumps({'token':tid,'selection_rule':'First token from predeclared (1000,50000,100000,200000) with zero accepted draft tokens at 64 and 128 outputs; validate again in every arm.','probes':rows},indent=2));return
 (D/'fixed-selection.json').write_text(json.dumps({'token':None,'probes':rows},indent=2));raise RuntimeError('No validated zero-acceptance control')

def suite(arm):
 token=json.loads((D/'fixed-selection.json').read_text())['token'];ids=prompt()
 run(arm+'-warm',ids,32,bias_token=token)
 for i in range(3):
  modes=('natural','fixed') if i%2==0 else ('fixed','natural')
  for mode in modes:run(f'{arm}-{mode}-{i}',ids,128,bias_token=token if mode=='fixed' else None)
 run(arm+'-natural-logprobs',ids,32,logprobs=10)
 ref=json.loads((D/'reference-outputs.json').read_text())
 for k in (4,5,6,15):run(f'{arm}-prefixA-{k}',ids+ref['A'][:k],1,logprobs=10)
 run(arm+'-prefixB-6',ids+ref['B'][:6],1,logprobs=10)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('arm');o=p.parse_args()
 with (D/(o.arm+'.gpu.csv')).open('w') as f:
  tele=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,nounits','-lms','500'],stdout=f,stderr=subprocess.DEVNULL)
  try:
   if o.arm=='probe':probe()
   else:suite(o.arm)
  finally:
   tele.terminate()
   try:tele.wait(timeout=5)
   except subprocess.TimeoutExpired:tele.kill();tele.wait()
