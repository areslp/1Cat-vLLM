"""Bounded planner pilot; exact eager forward and metadata graph checks."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics as st
import time
import traceback

import torch

P=argparse.ArgumentParser()
P.add_argument('--candidate',required=True)
P.add_argument('--output',required=True)
P.add_argument('--gpu',type=int,default=0)
P.add_argument('--mode',choices=('pilot','full'),default='pilot')
P.add_argument('--occupancy',required=True)
a=P.parse_args()
D=Path(a.output);D.mkdir(parents=True,exist_ok=False)
dev=torch.device('cuda',a.gpu);torch.cuda.set_device(dev)
ROOT=Path('/home/l/work/flash-next/perf-20260923-resume')
PROD=Path('/home/l/work/flash-next/prod-w48/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so')
OCCUPANCY_SHA='21504bcad9fcc7e6450bd789a4b832b9fe44cba25108ecddfc818ec4e9758fef'
PROD_SHA='a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b'
assert hashlib.sha256(Path(a.occupancy).read_bytes()).hexdigest()==OCCUPANCY_SHA
occupancy=json.loads(Path(a.occupancy).read_text())
expected_capture_hashes={row['path']:row['sha256'] for row in occupancy['captures']}
assert hashlib.sha256(PROD.read_bytes()).hexdigest()==PROD_SHA

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

refmod=load('flash_attn_v100.flash_attn_v100_cuda',PROD)
cand=load('qsa_planner58',a.candidate)
result={'gpu':a.gpu,'mode':a.mode,'status':'RUNNING','start':time.time(),'captures':[], 'synthetic':[], 'timing':[],
        'reference_so':str(PROD),'reference_sha256':hashlib.sha256(PROD.read_bytes()).hexdigest(),
        'candidate_sha256':hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest(),
        'forward':'unchanged original prod-w48 grouped_sparse_page4_split_fwd',
        'timing_basis':'complete binding includes hash init/union, scan, compact, bucket sort and scatter; no host count sync or extra kernel',
        'rank_mapping':'standalone physical GPU index; capture corpus not independent across four GPUs',
        'graph_scope':'planner metadata only; unchanged forward is compared eager; full forward graph/W1 admission pending cost gate',
        'occupancy_sha256':OCCUPANCY_SHA}

def persist():
    (D/'result.json').write_text(json.dumps(result,indent=2)+'\n')

def bits(t):
    return t.contiguous().view({torch.float16:torch.int16,torch.float32:torch.int32}[t.dtype])

def assert_equal(x,y,label,case):
    xx=bits(x) if x.is_floating_point() else x
    yy=bits(y) if y.is_floating_point() else y
    if not torch.equal(xx,yy):
        diff=(xx!=yy).nonzero();first=diff[0].tolist()
        torch.save({'case':case,'reference':x.cpu(),'candidate':y.cpu(),'label':label,'first':first},D/'first-divergence.pt')
        raise AssertionError(f'{label} bit divergence first={first},count={diff.shape[0]}')

def outputs(groups):
    return (torch.full((groups,4160),-777,dtype=torch.int32,device=dev),
            torch.full((groups,4160),0xA5A5A5A5,dtype=torch.uint32,device=dev),
            torch.full((groups,),-777,dtype=torch.int32,device=dev))

def call(mod,inp,out):
    fn=refmod.grouped_sparse_page4_plan_fwd if mod=='reference' else cand.plan_fwd
    fn(*inp[:5],*out,*inp[5:])

def metadata(inp,label):
    n=inp[0].shape[0]//8
    ro,co=outputs(n),outputs(n)
    call('reference',inp,ro);call('candidate',inp,co)
    torch.cuda.synchronize()
    for field,x,y in zip(('pages','masks','lengths'),ro,co):assert_equal(x,y,label+'/'+field,inp)
    return ro,co

def forward(q,pk,pv,plan,t2r,kvd,ks,vs):
    out=torch.full_like(q,float('nan'))
    lse=torch.full(q.shape[:2],float('nan'),device=dev,dtype=torch.float32)
    refmod.grouped_sparse_page4_split_fwd(q,pk,pv,out,*plan,lse,q.shape[2]**-.5,kvd,ks,vs,t2r)
    return out,lse

def graph_check(inp,label):
    ro,co=outputs(inp[0].shape[0]//8),outputs(inp[0].shape[0]//8)
    s=torch.cuda.Stream(device=dev);s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call('candidate',inp,co);call('reference',inp,ro)
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):call('candidate',inp,co)
    for rep in range(4):
        # Values change inside the existing graph addresses; flip input row order twice.
        if rep in (1,3):
            inp[0].copy_(torch.flip(inp[0],dims=[0]))
        call('reference',inp,ro);g.replay();torch.cuda.synchronize()
        for field,x,y in zip(('pages','masks','lengths'),ro,co):assert_equal(x,y,label+'/graph/'+field,inp)
    return {'replays':4,'same_shape_input_mutations':2,'pass':True}

def timed_pair(inp,label):
    buffers={key:outputs(inp[0].shape[0]//8) for key in ('reference','candidate')}
    graphs={};warm={}
    for key in buffers:
        s=torch.cuda.Stream(device=dev);s.wait_stream(torch.cuda.current_stream())
        start=time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(8):call(key,inp,buffers[key])
        torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
        warm[key]=(time.perf_counter()-start)*1000
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(16):call(key,inp,buffers[key])
        graphs[key]=g
    timings={mode:{key:[] for key in buffers} for mode in ('graph','eager')}
    host={key:[] for key in buffers}
    for mode in ('graph','eager'):
        for rep in range(16):
            for key in (('reference','candidate') if rep%2==0 else ('candidate','reference')):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                cpu=time.perf_counter();begin.record()
                if mode=='graph':graphs[key].replay()
                else:
                    for _ in range(16):call(key,inp,buffers[key])
                end.record();end.synchronize()
                timings[mode][key].append(begin.elapsed_time(end)*1000/16)
                if mode=='eager':host[key].append((time.perf_counter()-cpu)*1e6/16)
    out={'label':label,'rows':inp[0].shape[0],'warmup_ms':warm,'iterations_per_batch':16,'paired_repetitions':16,
         'samples_us':timings,'host_chain_us':host,'summary':{}}
    for mode,rows in timings.items():
        savings=[x-y for x,y in zip(rows['reference'],rows['candidate'])]
        out['summary'][mode]={'reference_us':st.median(rows['reference']),'candidate_us':st.median(rows['candidate']),
           'paired_saving_us':st.median(savings),'paired_saving_min_us':min(savings),'paired_saving_max_us':max(savings),
           'saving12calls_ms':12*st.median(savings)/1000}
    print(json.dumps({'timing':label,'summary':out['summary']}),flush=True)
    return out


def capture(path,do_graph=False,do_timing=False):
    c=torch.load(path,map_location='cpu',weights_only=True);m=c['meta']
    inp=[c[k].to(device=dev,dtype=dt).contiguous() for k,dt in
         [('logical_indices',torch.int32),('block_table',torch.int32),('token_to_req',torch.int32),
          ('query_positions',torch.int64),('sequence_lengths',torch.int32)]]
    inp += [m['k_shape'][1],m['k_stride'][0]//(4*c['q'].shape[2]),m['k_shape'][0]]
    label=path.parent.parent.name+'/'+path.name
    ro,co=metadata(inp,label)
    # Same compact KV bytes and unchanged original forward; remap all effective pages.
    ids=c['ids'].to(dev)
    q,pk,pv=(c[k].to(dev) for k in ('q','k_blocks','v_blocks'))
    def remap(plan):
        pages,masks,lengths=plan
        valid=torch.arange(pages.shape[1],device=dev)[None,:] < lengths[:,None]//4
        idx=torch.searchsorted(ids,pages.to(torch.int64).clamp(min=0)).clamp(max=ids.numel()-1)
        assert torch.equal(ids[idx][valid],pages.to(torch.int64)[valid]),'capture KV missing effective page'
        return (torch.where(valid,idx,torch.zeros_like(idx)).to(torch.int32),masks,lengths)
    rr=forward(q,pk,pv,remap(ro),inp[2],m['kv_cache_dtype'],m['k_scale'],m['v_scale'])
    cc=forward(q,pk,pv,remap(co),inp[2],m['kv_cache_dtype'],m['k_scale'],m['v_scale'])
    for field,x,y in zip(('out','lse'),rr,cc):assert_equal(x,y,label+'/'+field,c)
    for field,x in zip(('out','lse'),rr):assert_equal(x,c[field].to(dev),label+'/capture_reference/'+field,c)
    row={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tag':m['tag'],'kind':m['kind'],'meta':m,
         'rows':m['rows'],'full_metadata_exact':True,'forward_bits_exact':True,'reference_reproduces_capture':True}
    if a.mode=='pilot':
        assert row['sha256']==expected_capture_hashes[str(path)]
    if do_graph:row['graph']=graph_check(inp,label)
    if do_timing:
        timing=timed_pair(inp,label);timing.update(tag=m['tag'],kind=m['kind'],capture_meta=m)
        result['timing'].append(timing)
    result['captures'].append(row)
    persist()


def synthetic(seed,groups,variant):
    """Fixed-seed planner inputs within its page4 producer contract."""
    gen=torch.Generator().manual_seed(seed)
    nr=8;page_size=1632;ncache=64;stride=408;table_width=21
    li=torch.full((groups*8,2051),-1,dtype=torch.int32)
    bt=torch.randint(1,ncache,(nr,table_width),generator=gen,dtype=torch.int32)
    lengths=torch.tensor([0,1,1631,1632,1633,8192,32767,32768],dtype=torch.int32)
    mapping=torch.tensor([(i*3+seed)%nr for i in range(groups*8)],dtype=torch.int32)
    pos=torch.empty(groups*8,dtype=torch.int64)
    if variant=='shared_pages':bt[:]=bt[0].clone()
    if variant=='all_empty':mapping.fill_(-1)
    if variant=='slot_reuse':mapping=torch.flip(mapping,dims=[0])
    if variant=='null_page':bt[0].zero_();lengths[0]=1632
    if variant=='table_end':bt[:,-1]=-1
    for row,req in enumerate(mapping.tolist()):
        if req<0:pos[row]=-1;continue
        length=int(lengths[req]);visible=max(0,length-(row%4));pos[row]=visible-1
        count=min(visible//4,512)
        if count:
            pages=torch.randperm(max(1,length//4),generator=gen)[:count]
            if variant=='continuous':pages=torch.arange(count)
            if variant=='duplicates':pages=pages.remainder(max(1,count//3))
            if variant=='reverse':pages=pages.sort(descending=True).values
            li[row,:count*4]=(pages[:,None]*4+torch.arange(4)).reshape(-1).to(torch.int32)
        if visible%4:li[row,count*4]=visible//4*4
    inp=[x.to(dev).contiguous() for x in (li,bt,mapping,pos,lengths)]+[page_size,stride,ncache]
    label=f'seed{seed}-g{groups}-{variant}'
    ro,co=metadata(inp,label)
    pk=torch.randint(0,255,(ncache*stride,4,1,256),generator=gen,dtype=torch.uint8)
    # Exclude random E4M3 NaNs; isolated physical null microblocks stay poisoned.
    pk[(pk&127)==127]=126;pv=pk.roll(1,dims=0)
    pk[:stride]=127;pv[:stride]=255
    if variant=='signed_zero':pk[2*stride:3*stride]=0;pv[2*stride:3*stride]=128
    q=(torch.randn(groups*8,6,256,generator=gen)*.1).half().to(dev)
    pk,pv=pk.to(dev),pv.to(dev)
    rr=forward(q,pk,pv,ro,inp[2],'fp8_e4m3',1.,1.)
    cc=forward(q,pk,pv,co,inp[2],'fp8_e4m3',1.,1.)
    for field,x,y in zip(('out','lse'),rr,cc):assert_equal(x,y,label+'/'+field,inp)
    graph=graph_check(inp,label)
    result['synthetic'].append({'seed':seed,'groups':groups,'variant':variant,'rows':groups*8,
        'full_metadata_exact':True,'forward_bits_exact':True,'graph':graph})
    persist()

def boundary_input(count):
    """Legal binding stress input, not a real QSA selection capture."""
    nr=8;page_size=1632;ncache=24;stride=408;table_width=21
    li=torch.full((8,2051),-1,dtype=torch.int32)
    bt=torch.arange(1,table_width+1,dtype=torch.int32).expand(nr,-1).clone()
    mapping=torch.arange(nr,dtype=torch.int32)
    lengths=torch.full((nr,),32768,dtype=torch.int32)
    pos=lengths.to(torch.int64)-1
    remaining=count
    for row in range(nr):
        take=min(1024,remaining);remaining-=take
        # Distinct microblocks; no >8192-entry hash overflow or new input gate.
        li[row,:take]=4*(row*1024+torch.arange(take,dtype=torch.int32))
    assert remaining==0 and 0<=count<=8192
    inp=[x.to(dev).contiguous() for x in (li,bt,mapping,pos,lengths)]
    inp += [page_size,stride,ncache]
    return inp

def graph_bucket_transition():
    """One graph/address set across all runtime bucket branches, then empty."""
    counts=(0,1,1024,1025,2048,2049,4096,4097,8192,0,1)
    inp=boundary_input(1)
    ro,co=outputs(1),outputs(1)
    addresses=[x.data_ptr() for x in inp[:5]+list(ro)+list(co)]
    stream=torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call('reference',inp,ro);call('candidate',inp,co)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call('candidate',inp,co)
    rows=[]
    for count in counts:
        changed=boundary_input(count)
        for destination,source in zip(inp[:5],changed[:5]):
            destination.copy_(source)
        assert inp[5:]==changed[5:]
        # Clear every output before both calls, including inactive suffix bytes.
        for plan in (ro,co):
            plan[0].fill_(-777);plan[1].fill_(0xA5A5A5A5)
            plan[2].fill_(-777)
        call('reference',inp,ro);graph.replay();torch.cuda.synchronize()
        assert addresses==[x.data_ptr() for x in inp[:5]+list(ro)+list(co)]
        for field,x,y in zip(('pages','masks','lengths'),ro,co):
            assert_equal(x,y,f'bucket-transition-{count}/{field}',inp)
        rows.append({'expected_hash_entries':count,'full_metadata_exact':True})
        del changed
    result['metadata_graph_transition']={
        'one_graph':True,'fixed_input_output_addresses':addresses,
        'fresh_all_output_sentinels_each_replay':True,'cases':rows,
        'binding_stress_not_producer':True,'forward_graph_executed':False}
    persist()

def bucket_boundary(count):
    inp=boundary_input(count)
    page_size,stride,ncache=inp[5:]
    label=f'bucket-boundary-{count}'
    ro,co=metadata(inp,label)
    generator=torch.Generator().manual_seed(58000+count)
    pk=torch.randint(0,255,(ncache*stride,4,1,256),generator=generator,
                     dtype=torch.uint8)
    pk[(pk&127)==127]=126
    pv=pk.roll(1,dims=0);pk[:stride]=127;pv[:stride]=255
    q=(torch.randn(8,6,256,generator=generator)*.1).half().to(dev)
    pk,pv=pk.to(dev),pv.to(dev)
    rr=forward(q,pk,pv,ro,inp[2],'fp8_e4m3',1.,1.)
    cc=forward(q,pk,pv,co,inp[2],'fp8_e4m3',1.,1.)
    for field,x,y in zip(('out','lse'),rr,cc):
        assert_equal(x,y,label+'/'+field,inp)
    result['synthetic'].append({'case':label,'expected_hash_entries':count,
        'binding_stress_not_producer':True,'full_metadata_exact':True,
        'forward_bits_exact':True,'graph':graph_check(inp,label)})
    persist()


try:
    paths=sorted((ROOT/'step48-w0/cap48').glob('cap-*.pt'))+sorted((ROOT/'step48-w1cap/cap48').glob('qsa-*.pt'))
    assert len(paths)==194, f'real capture corpus changed: expected194 got{len(paths)}'
    seen=set()
    if a.mode=='pilot':
        chosen=[]
        for path in paths:
            c=torch.load(path,map_location='cpu',weights_only=True);m=c['meta'];key=(m['tag'],m['kind'],m['rows'])
            if m['kind']=='verify' and (key not in seen or m['tag']=='c8'):
                seen.add(key);chosen.append(path)
        paths=chosen;seen=set()
        assert len(paths)==33
        assert {str(path) for path in paths}==set(expected_capture_hashes)
    for index,path in enumerate(paths):
        c=torch.load(path,map_location='cpu',weights_only=True);m=c['meta'];key=(m['tag'],m['kind'],m['rows']);del c
        first=key not in seen;seen.add(key)
        capture(path,do_graph=first,do_timing=(first or m['tag']=='c8') and m['kind']=='verify')
        gc.collect()
    for count in (0,1,1024,1025,2048,2049,4096,4097,8192):
        bucket_boundary(count)
        gc.collect()
    graph_bucket_transition()
    if a.mode=='full':
        variants=('continuous','duplicates','reverse','shared_pages','all_empty','slot_reuse','table_end','null_page','signed_zero')
        for groups in range(1,9):
            for vi,variant in enumerate(variants):
                synthetic(541000+groups*100+vi,groups,variant)
    result['status']='PASS';result['independent_capture_files']=len(result['captures'])
    assert len(result['captures'])==len(paths)>0, 'real capture coverage incomplete'
    result['captured_rows']=sum(x['rows'] for x in result['captures'])
except BaseException:
    result['status']='FAIL';result['error']=traceback.format_exc()
finally:
    result['finished']=time.time();result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(dev);persist()
print(json.dumps({k:v for k,v in result.items() if k not in ('captures','synthetic','timing')}),flush=True)
raise SystemExit(result['status']!='PASS')
