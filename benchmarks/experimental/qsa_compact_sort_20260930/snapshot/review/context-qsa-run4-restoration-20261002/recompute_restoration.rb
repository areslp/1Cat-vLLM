# Common raw timing/fixed/concurrent/E7 checks copied from recompute_numeric7_restore.rb SHA 7b31f32d60e4f741f42bc69d5c4064e90fa6c54d98a025eabc6a1f515e0e698c
require 'json'
require 'digest'
BASE=File.expand_path(ARGV.fetch(0,'/private/tmp/flash-next-step58-20260930'))
REVIEW='review/context-qsa-run4-restoration-20261002'
OLD_API=166245
OLD_INV='e3ca051687274273a7c61b0861841142'
OLD_WORKERS=[166757,166758,166759,166760]
REMOTE='/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930'
PINS={}
def read(relative)
  path=File.join(BASE,relative)
  PINS[REMOTE+'/'+relative]=Digest::SHA256.file(path).hexdigest
  JSON.parse(File.read(path))
end
def check(value,name)
  raise name unless value
end
def median(values)
  a=values.sort;m=a.length/2
  a.length.odd? ? a[m] : (a[m-1]+a[m])/2.0
end
root='context-qsa-run4/restoration'
run='context-qsa-run4'
receipt=read(root+'/RESTORATION.json')
requests=read(root+'/original-check/requests-result.json')
ready=read(root+'/readiness.json')
before=read(root+'/original-check/before-snapshot.json')
after=read(root+'/original-check/after-snapshot.json')
cleanup=read(run+'/control/post-outer-cleanup.json')
budget=read(run+'/control/attempt1/artifact-post-outer.json')
live=read(REVIEW+'/FINAL-ORIGINAL-BINDING.json')
observed=read(REVIEW+'/FINAL-ORIGINAL-OBSERVED.json')
fresh=read(REVIEW+'/FINAL-ORIGINAL-SYSTEMD.json')
readback=read(REVIEW+'/FINAL-ORIGINAL-READBACK.json')
expected_pid=receipt['pid']
collector_path=File.join(BASE,REVIEW,'collect_final_original.py')
check(Digest::SHA256.file(collector_path).hexdigest=='603a33e73090c6662c8d8ab8b35477de13f25638d41cd954d63c1d26ae1f7a98','reviewed collector source SHA')
readback['pins'].each do |remote, sha|
  check(remote.start_with?(REMOTE+'/'),'collector pin root')
  relative=remote.delete_prefix(REMOTE+'/')
  path=File.join(BASE,relative)
  check(Digest::SHA256.file(path).hexdigest==sha,'collector inputs unchanged '+relative)
  PINS[remote]=sha
end
expected_gates=%w[controller_restore_exit readiness_maturity stable_pid no_restarts source_config_kv_swap_cleanup_worker original_unit requests six_fast_rounds all_24_fast fixed_exact finite_concurrency e7_rank_equal_compressed no_active_jobs no_guardian no_oom timer_cleared cgroup_oom_zero]
check(receipt['status']=='PASS' && receipt['gates'].keys.sort==expected_gates.sort && receipt['gates'].values.all?{|v|v==true},'strict named 17 gates')
check(expected_pid.is_a?(Integer) && expected_pid>0 && expected_pid!=OLD_API && live['api_pid']==expected_pid && readback['pid']==expected_pid,'actual new original PID')
check(readback['status']=='PASS_FINAL_ORIGINAL_INDEPENDENT_IDENTITY' && readback['no_unit_mutations']==true && readback['no_generation_requests']==true && readback['selected_API_flags']=='PASS_INDEPENDENT_RAW_PROC_READ','read-only raw API identity evidence')
inv=live['invocation_id']
check(inv.is_a?(String) && inv.match?(/\A[0-9a-f]{32}\z/) && inv!=OLD_INV && readback['invocation_id']==inv,'new InvocationID')
worker_pids=live['workers'].map{|w|w['pid']}
check(live['workers'].map{|w|w['rank']}==[0,1,2,3] && worker_pids.uniq.size==4 && worker_pids.all?{|p|p.is_a?(Integer) && p>0} && (worker_pids & (OLD_WORKERS+[OLD_API,expected_pid])).empty? && readback['workers']==worker_pids,'four new original workers')
[fresh,observed['systemd']].each do |props|
  check(props.values_at('MainPID','InvocationID','ActiveState','SubState','NRestarts')==[expected_pid.to_s,inv,'active','running','0'],'actual fresh systemd binding')
end
check(receipt['kv_tokens']==663816 && receipt['head']=='2dd437062a971a83d1f905249d288269c8daf501','original source and KV')
check(ready['status']=='READY_AND_MATURE' && ready['ready_maturity_s']>=360 && ready['pid']==expected_pid,'maturity')
check((ready['verified_epoch']-ready['http_ready_epoch']-ready['ready_maturity_s']).abs<0.001,'maturity recompute')
[before,after].each do |s|
  check(s['systemd']['MainPID']==expected_pid.to_s && s['systemd']['NRestarts']=='0','stable PID')
  check(s['workers'].sort_by{|w|w['rank']}.map{|w|w['pid']}==worker_pids,'closure four worker binding')
  check(s['queue'].values.all?(&:zero?) && s['checks'].reject{|k,_|k=='flags'}.values.all?,'snapshot gates')
  check(s['checks']['flags']==false,'preserve raw flag limitation')
end
check(requests['status']=='PASS' && requests['requests'].size==24 && requests['rounds'].size==6,'recovery coverage')
timings = requests['requests'].map do |entry|
  name = "round#{entry['rep']}-#{entry['mode']}-o#{entry['offset']}"
  raw = read("#{root}/original-check/#{name}.json")
  old = read("w1-run1/restoration/original-check/#{name}.json")
  old_cmp = read("w1-run1/restoration/original-check/#{name}-comparison.json")
  cmp = read("#{root}/original-check/#{name}-comparison.json")
  check(cmp == entry, 'comparison receipt binding')
  check(entry['reference_sha256'] == old_cmp['reference_sha256'] &&
        entry['reference'] == old_cmp['reference'], 'same historical reference')
  measured = median(raw['chunks'].map(&:first).drop(1).each_cons(2).map do |x, y|
    1000 * (y - x)
  end)
  check((measured - entry['step_ms']).abs < 1e-9, 'raw SSE timing')
  check(measured <= entry['reference_step_ms'] * 1.01 &&
        (entry['mode'] == 'G1' || measured < 34), 'individual fast gate')
  check(raw['ok'] && raw['checks'].values.all? && entry['checks'].values.all?,
        'request correctness')
  check(raw['usage'].values_at('prompt_tokens', 'completion_tokens') == [512, 256],
        'token usage')
  check(raw['output_token_ids'] == old['output_token_ids'] &&
        raw['output_token_ids'].length == 256, 'tokens vs prior restored original')
  keys = raw['metrics_delta'].keys.select { |k| k.start_with?('vllm:spec_decode_') }
  check(keys.length == 3 && keys.all? { |k| raw['metrics_delta'][k] == old['metrics_delta'][k] },
        'acceptance vs prior restored original')
  {'rep'=>entry['rep'], 'mode'=>entry['mode'], 'step_ms'=>measured}
end
requests['rounds'].each do |round|
  rows = timings.select { |r| r['rep'] == round['rep'] && r['mode'] == round['mode'] }
  check(rows.length == 4 &&
        (median(rows.map { |r| r['step_ms'] }) - round['median_ms']).abs < 1e-9,
        'round median recompute')
  check(round['fast_count'] == 4 && round['slow_count'] == 0 && round['pass'],
        'round verdict')
end
%w[natural full mask].each do |mode|
  raw = read("#{root}/original-check/fixed-#{mode}.json")
  old = read("w1-run1/restoration/original-check/fixed-#{mode}.json")
  check(raw['output_token_ids'] == old['output_token_ids'] &&
        raw['logprobs'] == old['logprobs'], 'fixed JSON values vs prior restored original')
end
check(requests['e7_pass'] && requests['concurrent'].length == 4 &&
      requests['concurrent'].all? { |r| r['checks'].values.all? }, 'E7/concurrency')
check(requests['concurrent'].map { |r| [r['n'], r['rep']] } ==
      [[4,0],[4,1],[8,0],[8,1]], 'exact four finite concurrency groups')
requests['concurrent'].each do |entry|
  raw = read("#{root}/original-check/matched-N#{entry['n']}-#{entry['rep']}.json")
  check(raw['checks'].values.all? && raw['rows'].length == entry['n'] &&
        raw['metrics_delta'] == entry['metrics_delta'], 'raw concurrency coverage')
  raw['rows'].each do |row|
    check(row['ok'] && row['output_token_ids'].length == 256 &&
          row['usage'].values_at('prompt_tokens','completion_tokens') == [512,256],
          'every concurrent request finished exactly')
  end
end
%w[G1 N1].each do |mode|
  raw = read("#{root}/original-check/warmup-#{mode}.json")
  check(raw['ok'] && raw['checks'].values.all? &&
        raw['usage'].values_at('prompt_tokens','completion_tokens') == [512,256],
        'both original warmups complete')
end
check(2 + requests['requests'].length +
      requests['concurrent'].sum { |r| r['n'] } + requests['fixed'].length == 53,
      'complete original 53-request restoration packet')
fences = %w[before after].map do |tag|
  fence = read("#{root}/original-check/#{tag}-concurrent-fence.json")
  snapshot = read("#{root}/original-check/#{tag}-concurrent-snapshot.json")
  first, second = fence.values_at('first', 'second')
  check(fence['status'] == 'FRESH_STABLE_FULL_COUNTERS_ALL_RANKS' &&
        fence['boundary_epoch'] >= fence['last_completed_epoch'] &&
        fence['export_period_s'] == 1.0 && fence['passive_budget_s'] == 15,
        'passive fresh fence boundary')
  check(first.map { |r| r['rank'] } == [0,1,2,3] &&
        second.map { |r| r['rank'] } == [0,1,2,3] &&
        (first + second).map { |r| r['counters'] }.uniq.size == 1,
        'all-rank complete counter equality')
  first.zip(second, after['workers']).each do |a, b, reference|
    check(a['time'] > fence['boundary_epoch'] && b['time'] - a['time'] >= 1.0,
          'fresh publication spans one period')
    %w[pid rank ranks mode module_path package_hashes].each do |key|
      check(a[key] == b[key] && a[key] == reference[key], 'fence source/PID binding')
    end
  end
  check(snapshot['workers'].map { |r| r['counters'] } == second.map { |r| r['counters'] },
        'snapshot full vectors unchanged after fence')
  fence
end
strict_delta = fences[0]['second'].zip(fences[1]['second']).map do |a, b|
  # Pinned runtime initializes collections.Counter() and exports dict(_COUNTS).
  # These two keys are first created by the concurrent workload, at zero plus
  # their first increment. Preserve all keys and reject removed/unknown ones.
  added = b['counters'].keys - a['counters'].keys
  check((a['counters'].keys - b['counters'].keys).empty? &&
        (added - %w[baseline_local_reused fallback_structure]).empty?,
        'source-declared sparse Counter key set')
  {'rank'=>b['rank'], 'pid'=>b['pid'], 'counters'=>b['counters'].to_h { |k, v| [k, v-a['counters'].fetch(k, 0)] }}
end
counter_source = File.join(BASE, 'review/service3-e7-runtime.production.py')
counter_sha = Digest::SHA256.file(counter_source).hexdigest
check(counter_sha == '6c901559b998c3777fbcde77f94f373d8ad13327d50b4b2f352ea94266bb901d' &&
      after['workers'].all? { |r| r['package_hashes']['runtime.py'] == counter_sha },
      'actual sparse Counter source matches all exporters')
PINS["#{REMOTE}/review/service3-e7-runtime.production.py"] = counter_sha
PINS['/home/l/work/1Cat-vLLM-kv-w12/vllm/v1/worker/gpu/sample/sm70_e7/runtime.py'] = counter_sha
check(strict_delta.map { |r| r['counters'] }.uniq.size == 1 &&
      strict_delta.all? { |r| r['counters'].values.all? { |v| v >= 0 } && r['counters']['compressed_steps'] > 0 },
      'strict E7 full delta independently recomputed')
check(strict_delta == requests['e7_delta'], 'raw complete delta agrees with worker receipt')
check(cleanup['status']=='PASS' && cleanup['gates'].values.all?{|v|v==true},'post outer cleanup')
check(cleanup['GPU_processes'].size==4 && cleanup['GPU_processes'].all?{|r|r['in_original_cgroup']==true} && cleanup['GPU_processes'].map{|r|r['pid']}.sort==worker_pids.sort,'actual four original GPU ownership')
check(budget['status']=='PASS' && budget['measured_bytes']<=budget['threshold_bytes'] && budget['threshold_bytes']==8*1024**3 && budget['hard_FS_quota']==false,'measured budget')
strict_exits=%w[restoration/verify.exit restoration/evaluate.exit control/post-outer.exit control/artifact-final.exit]
experiment_exits=%w[control/window.exit control/outer-launch.exit]
optional_exits=%w[control/ROOT-LAUNCH-EXIT.txt control/attempt1/outer-cleanup.exit control/attempt1/outer-start.exit]
native_exits={}
(strict_exits+experiment_exits+optional_exits).each do |relative|
  path=File.join(BASE,run,relative)
  if !File.exist?(path) && optional_exits.include?(relative)
    native_exits[relative]=nil
    next
  end
  PINS[REMOTE+'/'+run+'/'+relative]=Digest::SHA256.file(path).hexdigest
  text=File.read(path).strip
  check(text.match?(/\A[0-9]+\z/),'terminal integer exit '+relative)
  code=Integer(text)
  check((0..255).include?(code),'bounded terminal exit '+relative)
  check(code==0,'strict restoration exit '+relative) if strict_exits.include?(relative)
  native_exits[relative]=code
end
check(native_exits==readback['native_exits'],'readback terminal exit binding')
analysis_path=File.join(BASE,run,'analysis/RESULT.json')
analysis_exit_path=File.join(BASE,run,'control/attempt1/analysis.log.exit.json')
analysis=File.exist?(analysis_path) ? read(run+'/analysis/RESULT.json') : nil
analysis_exit=File.exist?(analysis_exit_path) ? read(run+'/control/attempt1/analysis.log.exit.json') : nil
if analysis
  check(analysis_exit && analysis['diagnostic_exit']==analysis_exit['exit'],'native analysis decision binding')
end
experiment_status=if experiment_exits.any?{|name|native_exits[name]!=0}
  'NONZERO_EXPERIMENT_OR_OUTER_RETAINED'
else
  'ZERO_WINDOW_AND_OUTER'
end
check(experiment_status==readback['experiment_exit_status'],'experiment status preservation')
report={status:'PASS_PARENT_RECOMPUTED_ORIGINAL_53_RESTORATION',pid:expected_pid,invocation_id:live['invocation_id'],workers:live['workers'].map{|w|w['pid']},gates:receipt['gates'],requests:53,fast:24,slow:0,rounds:requests['rounds'],readiness:ready,raw_SSE_recomputed:true,exact_tokens_and_acceptance_crosschecked:true,fixed_JSON_logprobs_crosschecked:true,raw_FP32_bits_claimed:false,E7_full_delta:strict_delta,raw_flags_boolean:false,selected_API_flags:'separately matched by read-only /proc collector',cleanup:cleanup['gates'],artifact_budget:budget,native_exits:native_exits,experiment_exit_status:experiment_status,experiment_analysis:analysis,experiment_analysis_native_exit:analysis_exit,pins:PINS,permanent_deployment:false,push:false,merge:false}
out=File.join(BASE,REVIEW,'PARENT-RESTORATION-REVIEW.json')
File.write(out,JSON.pretty_generate(report)+"\n",mode:'wx')
puts JSON.generate(report.reject{|k,_|k==:pins})
