"""Local create-only review packet generator; stdlib only, no model imports."""
import ast
import json
from pathlib import Path
from io_contract import save, sha256
from make_config import create, SOURCES, K
from policy import TARGET_NAMES, DESCRIPTORS

ROOT = Path(__file__).resolve().parent
D = '/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930'
SERVICE = D + '/service-implementation'
PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
E = ROOT/'evidence'
INVENTORY = E/'cohort-inventory.input.json'
inventory = json.loads(INVENTORY.read_text())
# Examples are review artifacts, not an executable controller plan.
for mode in ('off','shadow'):
    config = create(inventory, mode, D+f'/service-run1/{mode}/capture', [], mode=='off')
    config['inventory_path'] = SERVICE+'/evidence/cohort-inventory.input.json'
    config['inventory_sha256'] = sha256(INVENTORY)
    save(ROOT/f'config-{mode}.template.example.json', config)
source_rows = [{'key': key, 'snapshot_path': 'source/'+file,
                'sha256': sha256(ROOT/'source'/file),
                'runtime_path': path if path.startswith('/') else K+'/'+path}
               for key,(file,path) in SOURCES.items()]
contract = {
 'status':'SOURCE_CPU_REVIEW_PACKET_NOT_SERVICE_ADMISSION',
 'candidate_binary':{'path':D+'/build/candidate1/qsa_planner58.so',
  'sha256':'e5ac0b418ebb9a387e0838b230dd15114185f637804e9d14b336897ab24bb7e3'},
 'original_binary':{'path':'/home/l/work/flash-next/prod-w48/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so',
  'sha256':'a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b'},
 'source_pins':source_rows, 'owners':list(TARGET_NAMES),
 'draft_excluded':'mtp.layers.48.self_attn.attn',
 'descriptors':[{'target_descriptor':key,'grouped_rows':rows} for key,rows in sorted(DESCRIPTORS.items())],
 'real_consumers':[4,5,6,7,8], 'excluded':'c1-c3/nonuniform/prefill/draft/PIECEWISE/eager/unsupported/unreviewed bucket',
 'formal_capture_line':378,'capture_order_source_line':301,
 'modes':{'off':'candidate loaded, original planner; default no runtime observers/GPU buffers',
  'shadow':'original model outputs + independent private candidate planner/original forward + exact device checks',
  'on':'only candidate planner graph node; original floating forward + tail; no comparison/counter/bank/observers'},
 'diagnostic_host_audit':'off only, explicit admission instrumented flag; never W2 performance arm',
 'epoch_count':20, 'drain_count':20,
 'inventory_path':'evidence/cohort-inventory.input.json','inventory_sha256':sha256(INVENTORY),
 'http_inventory_scope':'20 exact cohort IDs/arm only; actual bodies/prompts/wall/cap owned and separately frozen by control',
 'counts':inventory['counts'],'total_http_in_inventory':296,'http_cap_controller':320,
 'epoch_gate':{'eligible':'required actual c; scheduled5/draft4 + exact approved target cached replay and device counts',
  'c1_c3':'original actual short uniform route; no eligible dispatch/replay/hit',
  'mixed':'same primary target dispatch actual>=2/uniformNone/prefill scheduled>1 draft0 plus verify scheduled5 draft4',
  'fallback':'all primary target dispatch original/uniformNone; actual1..declared c; no eligible replay/hit',
  'global_shape_guard':'fallback two groups at least one actual>=4, all scheduled5/draft0/uniformNone',
  'slotreuse':'actual finished old slot_remove -> distinct new slot_add same slot -> physical block_append; no immutable KV claim'},
 'arm_schema':{'epoch':'exact registered epoch','external_request_ids':'exact main list'},
 'drain_arm_schema':{'id':'same registered epoch','epoch':'same','cohort_receipt':'exact file under cfg.cohort_dir',
   'cohort_sha256':'client receipt hash'},
 'drain_protocol':'complete primary HTTP+queueempty->persistclient->armdrain->one predeclared c1 sentinel->four drains->immediate gate->remove old arms before next',
 'sentinel':'declared separately; excluded hit/fallback/mixed/consumer proof; actual ID mapping required',
 'identity_helper':{'path':'request_ids.py','sha256':sha256(ROOT/'request_ids.py')},
 'scheduler_ids':'strict external-[0-9a-f]{8} bijection; preserve original randomization; n=1/singleprompt/nonstream',
 'bounds':{'workspace_bytes_per_rank':32*1024**2,'first_bank_bytes_per_rank':64*1024**2,
   'first_artifact_bytes_per_rank':64*1024**2,'added_allocated_vs_off_bytes_per_rank':256*1024**2,
   'added_reserved_vs_off_bytes_per_rank':512*1024**2,'lifecycle_events_per_rank':4096,
   'lifecycle_bytes_per_rank':4*1024**2,'host_events_per_epoch_rank':4096,
   'drain_json_bytes':8*1024**2,'shadow_or_off_epoch_seconds':600,'required_kv_tokens':663816},
 'bank_bytes_formula':'770472 + 4*block_table.numel + 4*sequence_lengths.numel + 32*prepared_nodes (q40x6x256/topk40x416)',
 'first_divergence':'immutable bank first flag.node_index; separate lowest_bad_counter_row; raw integer snapshots',
 'reproducibility':'integer planner input complete incl host scalars; standalone float replay lacks KV values',
 'baseline_generator':{'actual':'baseline.py main; control wrapper prevalidates off startup/cgroup/NVML and calls this sole create-only generator',
  'binding_fields':['rank','path','sha256','pid','config_sha256','candidate_binary_sha256','original_binary_sha256','device_uuid','kv_num_blocks'],
  'uuid':'Torch raw string; NVML GPU- normalization only independent controller check',
  'allowed_changed_fields':['baseline_off_receipts'],
  'receipt_status':'PASS_ONLY_BASELINE_BINDINGS_DERIVED'},
 'unverified':['actualCLI/deferredstartup/currentmodelowners','real synthetic CUDAgraph smoke',
   'real service same-state shadow outputs/cached device hits/fallback/lifecycle',
   'candidate-on model numerics/teacherforcing','W2 SSE/stability/performance']}
save(ROOT/'contract.json',contract)
command = {'status':'REVIEWABLE_NOT_EXECUTION_APPROVAL','cwd':SERVICE,
 'install':{'shim_directory':SERVICE,'env':{'STEP58_SERVICE_CONFIG':'controller actual config path',
   'STEP58_RAW_GENERATION_PATH':'study arm private raw-generation.jsonl'},
   'parent_wrapper':'original run-flash-next-w12.sh with exact actual original env/profile/NUMA/compile/SO/marker preserved; only study PORT/HOST/D/stats/path changes',
   'config_ready':'capture-ready-rank{0..3}.json bound to actual cgroupworker PID/configSHA/source/library/UUID/KV'},
 'baseline':{'argv':[PY,SERVICE+'/baseline.py','--template','<frozen-shadow-template>',
   '--bindings','<four-exact-off-bindings>','--output','<create-only-shadow.runtime.json>',
   '--receipt','<create-only-service-derivation.json>'],'timeout_seconds':30},
 'per_cohort':{'argv':[PY,SERVICE+'/admission.py','--ready-dir','<arm/capture>',
   '--drain-id','<registered epoch>','--output','<cohort/admission-gate.json>'],
   'eligible_shadow_extra':['--require-hit'],'timeout_seconds':30,
   'status_off':'OFF_HOST_COHORT_GATES_PASS_DEVICE_UNINSTRUMENTED',
   'status_shadow':'SHADOW_COHORT_GATES_PASS_BENCHMARK_UNVERIFIED',
   'result':'per-cohort ranks[4] and epochs[1].ranks[4]; full actual consumer/mapping/route/slot proof',
   'failure_status':'STOP_SERVICE_GATE','pass_exit':0,'failure_exit':2},
 'all_epochs':{'argv':[PY,SERVICE+'/admission.py','--ready-dir','<arm/capture>',
   '--all-epochs','--require-slot-reuse','--output','<arm/all-epochs-gate.json>'],
   'timeout_seconds':30,'result':'epochs[20], each 4ranks; fallback shape-only guard + slot reuse required'},
 'cpu_gate_env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
   'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','remove_env_prefix':'STEP58_'},
 'smoke':'evidence/GPU-SMOKE-PROPOSAL.final.json required before first service load only after parent maintenance-window approval; one diagnostic no rerun',
 'full_service_window':'controller owns exact unit/timer/runtime config/shadow cohorts/immediate gates/restore; no on or W2 from this contract'}
save(ROOT/'command-contract.json',command)
smoke={'status':'PROPOSED_NOT_EXECUTED_PARENT_GPU_GATE_REQUIRED',
 'required_before_service_shadow':True,'timeout_seconds':60,'inner_timeout_seconds':55,
 'kill_after_seconds':5,'RuntimeMaxSec':60,'KillMode':'control-group',
 'MemoryMax':2*1024**3,'MemorySwapMax':0,'User':'l','WorkingDirectory':SERVICE,
 'CUDA_VISIBLE_DEVICES':'GPU-54a5dec0-85a9-837b-c54d-4a3752c1b620',
 'uuid_provenance':'prior closed W1 GPU0 identity; controller fresh device validation required',
 'argv':['/usr/bin/timeout','--kill-after=5s','55','/usr/bin/env',
  'CUDA_VISIBLE_DEVICES=GPU-54a5dec0-85a9-837b-c54d-4a3752c1b620',
  'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
  'OPENBLAS_NUM_THREADS=1','/usr/bin/taskset','-c','14','/usr/bin/nice','-n','15',
  PY,SERVICE+'/smoke_graph.py','--device','0','--output',D+'/service-run1/smoke1/RESULT.json'],
 'allocated_gpu_cap':64*1024**2,'reserved_gpu_cap':128*1024**2,
 'nvml_process_total_proposed_cap':768*1024**2,'nvml_context':'controller must measure actual process context separately; allocator caps exclude CUDA context',
 'artifact_cap':64*1024**2,'source_files':[{'path':f,'sha256':sha256(ROOT/f)}
   for f in ('smoke_graph.py','first_failure.py','shadow.py','io_contract.py')],
 'checks':['rawNaN/signedzero','false_predicate','firstfail','secondfail_no_overwrite','safe_drain','reset','guards','alias'],
 'result':'PASS_SYNTHETIC_SHADOW_GRAPH_NOT_SERVICE','not_proved':['actual target/candidate library integration','real KV attention replay']}
# Preserve prior unexecuted proposal; new file is authoritative.
save(E/'GPU-SMOKE-PROPOSAL.final.json',smoke)
for path in ROOT.rglob('*.py'):
    if 'uv-cache' not in path.parts: ast.parse(path.read_text(),filename=str(path))
receipt={'status':'PASS_SOURCE_AND_PROTOCOL_CPU_NOT_SERVICE',
 'local_runtime':{'python':'uv-managed CPython3.12.13','uv':'0.11.16',
   'CUDA_VISIBLE_DEVICES':'','torch':'not imported by protocol/source tests'},
 'checks':[{'path':'evidence/cpu-tests-attempt1.log','result':'retained initial NONE mock failure'},
  {'path':'evidence/cpu-tests-attempt2.log','result':'12PASS'},
  {'path':'evidence/cpu-protocol-attempt1.log','result':'4PASS'},
  {'path':'evidence/cpu-protocol-attempt2.log','result':'7PASS'},
  {'path':'evidence/cpu-route-final.log','result':'1 related final mixed predicate PASS'},
  {'path':'evidence/cpu-capture-order.json','result':'actual AST sort/strict order guard PASS'},
  {'path':'evidence/CPU-TORCH-RECEIPT.json','result':'8 synthetic realTorch CPU PASS, orchestration defect preserved; configured cap not measured RSS'}],
 'actual_executed_source':'evidence/cpu-torch-attempt1/executed-source-provenance.json',
 'scope':'No repeated Torch run, GPU/HTTP/service requests, tensor scans, installs, production changes',
 'failure_notes':['preexecution owner suffix correction','source gate deferred install',
   'metadata lengths tokens vs pages checker defect corrected',
   'total graph memory mistaken as incremental candidate fixed',
   'zero-token lifecycle loss fixed','arrival count vs dispatch predicate fixed',
   'first_bad_node chronological flag vs lowest counter row fixed']}
for row in receipt['checks']: row['sha256']=sha256(ROOT/row['path'])
save(E/'CPU-RECEIPT.final.json',receipt)
files=[]
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or any(x in path.parts for x in ('uv-cache','__pycache__')): continue
    rel=path.relative_to(ROOT).as_posix()
    if rel in ('manifest.json','upload-files.txt') or path.suffix in ('.pyc','.tmp'): continue
    assert not path.is_symlink()
    files.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha256(path)})
save(ROOT/'manifest.json',{'status':'FROZEN_SOURCE_CPU_PACKET_V1_NOT_SERVICE_ADMISSION',
    'files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files)})
with (ROOT/'upload-files.txt').open('x') as stream:
    stream.write('\n'.join([x['path'] for x in files]+['manifest.json','upload-files.txt'])+'\n')
print(json.dumps({'manifest_sha256':sha256(ROOT/'manifest.json'),'files':len(files),
 'contract_sha256':sha256(ROOT/'contract.json'),'command_sha256':sha256(ROOT/'command-contract.json'),
 'bytes':sum(x['bytes'] for x in files),'status':'REVIEW_PACKET_FROZEN'}))
