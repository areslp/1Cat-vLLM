"""Create-only retry2 packet; canonical command assignment is idempotent."""
import ast
import copy
import json
from pathlib import Path

from io_contract import save, sha256
from make_config import create

ROOT = Path(__file__).resolve().parent
D = '/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930'
CANONICAL = D + '/service-implementation-retry2'
PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
E = ROOT / 'evidence'
PRIOR = ROOT.parent / 'service-implementation-retry1'
CORE = ('admission.py', 'baseline.py', 'capture_site.py', 'cohort.py', 'domains.py',
        'first_failure.py', 'identity.py', 'io_contract.py', 'make_config.py',
        'policy.py', 'request_ids.py', 'shadow.py', 'smoke_graph.py', 'sitecustomize.py')


def canonical_command(value):
    value = copy.deepcopy(value)
    value['cwd'] = CANONICAL
    value['install']['shim_directory'] = CANONICAL
    for key, name in (('all_epochs','admission.py'),('per_cohort','admission.py'),
                      ('baseline','baseline.py')):
        value[key]['argv'][0:2] = [PY, CANONICAL + '/' + name]
        assert (ROOT / name).is_file()
    value['cpu_corrective'] = {
        'status':'COMPLETED_5PASS_NO_AUTORERUN', 'cwd':CANONICAL,
        'plan':'evidence/cpu-serialization-attempt2-plan.json',
        'result':'evidence/cpu-serialization-attempt2/results/RESULT.json',
        'timeout_seconds':30,'scope':'real NumPy source AST + full wrapper/drain; GPU explicit stubs'}
    return value


def main():
    existing_manifest = ROOT / 'manifest.json'
    if existing_manifest.exists():
        assert sha256(existing_manifest) == sha256(PRIOR/'manifest.json'), (
            'create-only freeze: current packet already produced')
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text())
    for row in prior_manifest['files']:
        assert sha256(PRIOR / row['path']) == row['sha256'], 'retry1 must remain immutable'
    core = {name:{'sha256':sha256(ROOT/name), 'identical_to_retry1':True}
            for name in CORE}
    assert all(sha256(ROOT/name)==sha256(PRIOR/name) for name in CORE)
    for name in ('manifest.json','command-contract.json','contract.json',
                 'config-off.template.example.json','config-shadow.template.example.json',
                 'upload-files.txt'):
        path=ROOT/name
        if path.exists():
            assert sha256(path)==sha256(PRIOR/name), 'only inherited metadata may move'
            assert not (E/('inherited-retry1-'+name)).exists(), 'preserve prior metadata'
            path.rename(E/('inherited-retry1-'+name))
    command=canonical_command(json.loads((PRIOR/'command-contract.json').read_text()))
    assert canonical_command(command)==command
    assert command['cwd']==command['install']['shim_directory']==CANONICAL
    assert command['cpu_corrective']['cwd']==CANONICAL
    for key in ('all_epochs','per_cohort','baseline'):
        argv=command[key]['argv']
        assert argv[0]==PY and str(Path(argv[1]).parent)==CANONICAL
        assert (ROOT/Path(argv[1]).name).is_file()
        assert '-retry1-retry1' not in argv[1] and '-retry2-retry2' not in argv[1]
    command['status']='FROZEN_RETRY2_SOURCE_CPU_NOT_EXECUTION_APPROVAL'
    command['smoke']='PRIOR_SYNTHETIC_BANK_PROOF_REUSE_ONLY; first_failure/shadow/smoke_graph/io_contract identical; no rerun from this packet'
    contract=json.loads((PRIOR/'contract.json').read_text())
    contract['status']='SOURCE_CPU_RETRY2_EVENT_BOUNDARY_PASS_NOT_SERVICE_ADMISSION'
    contract['packet_generator']='freeze_retry2.py; create-only artifacts, canonical assignment'
    contract['prior_retry1']=contract.pop('corrective_retry')
    contract['corrective_retry']={
        'review':'RETRY2-SOURCE-REVIEW.md', 'status':'5_REAL_NUMPY_CPU_PASS_GPU_SERVICE_NOT_RUN',
        'cause':'other-manager dispatch uniform np.int32 from Eagle array.max/get_uniform/DP1 passthrough',
        'runtime_delta':'Runtime.event normalizes all four producer schemas',
        'strict_integer':'exact Python int/np.integer only; rejectbool/float/array/tensor/custom; None only optional',
        'runtime_loader_source_pins_13_unchanged':True,
        'retry1_manifest_sha256':sha256(PRIOR/'manifest.json')}
    contract['supplemental_source_evidence'].update({key:
        {'local':'source/'+file, 'sha256':sha256(ROOT/'source'/file),
         'path':'/home/l/work/1Cat-vLLM-kv-w12/'+runtime}
        for key,file,runtime in (
            ('eagle','eagle_speculator.production.py','vllm/v1/worker/gpu/spec_decode/eagle/speculator.py'),
            ('dp','dp_utils.production.py','vllm/v1/worker/gpu/dp_utils.py'))})
    inventory=json.loads((E/'cohort-inventory.input.json').read_text())
    for mode in ('off','shadow'):
        cfg=create(inventory,mode,CANONICAL+f'/evidence/config-example-{mode}/capture',[],mode=='off')
        cfg['inventory_path']=CANONICAL+'/evidence/cohort-inventory.input.json'
        cfg['inventory_sha256']=sha256(E/'cohort-inventory.input.json')
        cfg['baseline_off_capture_dir']=CANONICAL+'/evidence/config-example-off/capture'
        save(ROOT/f'config-{mode}.template.example.json',cfg)
    save(E/'RETRY2-UNCHANGED-MECHANISM.json',{
        'status':'14_CORE_FILES_IDENTICAL_PRIOR_BANK_SMOKE_REUSE_NOT_NEW_SERVICE_PROOF',
        'files':core, 'candidate_binary':contract['candidate_binary'],
        'original_binary':contract['original_binary'],
        'prior_smoke_result':D+'/service-run1/smoke1/RESULT.json',
        'prior_smoke_result_sha256':'0ddc7ee9e676f2d016b3b8444f2a0a4a41cf977f32fe7b6019e18f8a4a869394'})
    result=E/'cpu-serialization-attempt2/results/RESULT.json'
    assert json.loads(result.read_text())['status']=='PASS_SOURCE_NUMPY_WRAPPER_DRAIN_NOT_SERVICE'
    save(E/'CPU-SERIALIZATION-RECEIPT.json',{
        'status':'PASS_RETRY2_SOURCE_CPU_ONLY','test_count':5,
        'attempt1':'exit1/3errors stage dependency missing; not event-chain evidence',
        'attempt2':'exit0/5PASS code unchanged; dependencies corrected once after parent review',
        'result':{'path':str(result.relative_to(ROOT)),'sha256':sha256(result)},
        'log':{'path':'evidence/cpu-serialization-attempt2.log',
               'sha256':sha256(E/'cpu-serialization-attempt2.log')},
        'code':{name:sha256(ROOT/name) for name in
                ('service_hook.py','event_contract.py','test_retry_serialization.py','cpu_serialization.py')},
        'bounds':'timeout30s/CUDAhidden/CPU14/nice15/threads1/noTorch/model/PT/GPU/API',
        'not_proved':['service admission','candidate float/graph behavior','W2 performance']})
    for path in ROOT.rglob('*.py'):
        if 'uv-cache' not in path.parts: ast.parse(path.read_text())
    names=[]
    for path in sorted(ROOT.rglob('*')):
        if (not path.is_file() or any(part in ('uv-cache','__pycache__') for part in path.parts)
                or path.suffix in ('.pyc','.tmp')): continue
        assert not path.is_symlink()
        rel=path.relative_to(ROOT).as_posix()
        if rel not in ('manifest.json','command-contract.json','upload-files.txt'):
            names.append(rel)
    names += ['command-contract.json','manifest.json']
    command['upload_files']=names
    assert len(names)==len(set(names))
    save(ROOT/'contract.json',contract)
    # contract was newly created after inventory; add exactly once.
    if 'contract.json' not in names: names.insert(0,'contract.json')
    command['upload_files']=names
    save(ROOT/'command-contract.json',command)
    files=[{'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha256(ROOT/name)}
           for name in names if name!='manifest.json']
    save(ROOT/'manifest.json',{
        'status':'FROZEN_RETRY2_SOURCE_CPU_5PASS_NOT_SERVICE_ADMISSION',
        'files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),
        'retry1_manifest_sha256':sha256(PRIOR/'manifest.json')})
    with (ROOT/'upload-files.txt').open('x') as f: f.write('\n'.join(names)+'\n')
    print(json.dumps({'manifest_sha256':sha256(ROOT/'manifest.json'),
        'command_sha256':sha256(ROOT/'command-contract.json'),
        'contract_sha256':sha256(ROOT/'contract.json'),
        'files':len(files),'upload_files':len(names),
        'total_bytes':sum(x['bytes'] for x in files)},sort_keys=True))


if __name__=='__main__':
    main()
