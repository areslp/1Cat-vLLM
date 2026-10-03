require 'json'
require 'digest'

BASE = File.expand_path('../..', __dir__)
REMOTE = '/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930'
PLAN_SHA = '355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4'
PINS = {}
def read_json(path)
  PINS[path.sub(BASE + '/', '')] = Digest::SHA256.file(path).hexdigest
  JSON.parse(File.read(path))
end
def require_check(value, message)
  raise message unless value
end
def local_path(path)
  require_check(path.start_with?(REMOTE + '/'), 'unexpected remote path')
  BASE + path.delete_prefix(REMOTE)
end
def tokens_from_raw(ref)
  path = local_path(ref.fetch('path'))
  require_check(File.size(path) == ref.fetch('bytes'), 'raw size differs')
  digest = Digest::SHA256.file(path).hexdigest
  require_check(digest == ref.fetch('sha256'), 'raw SHA differs')
  PINS[path.sub(BASE + '/', '')] = digest
  tokens = []
  done = 0
  File.foreach(path) do |line|
    item = JSON.parse(line)
    wire = [item.fetch('raw_hex')].pack('H*')
    next if wire.empty?
    require_check(wire.start_with?('data: '), 'unexpected SSE frame')
    payload = wire.delete_prefix('data: ').strip
    if payload == '[DONE]'
      done += 1
      next
    end
    packet = JSON.parse(payload)
    packet.fetch('choices').each do |choice|
      require_check(choice.fetch('index') == 0, 'unexpected choice')
      tokens.concat(choice.fetch('token_ids', []))
    end
  end
  require_check(done == 1 && tokens.length == 256, 'incomplete raw main stream')
  tokens
end

plan = BASE + '/context-qsa-run4/control/plan.json'
require_check(Digest::SHA256.file(plan).hexdigest == PLAN_SHA, 'wrong plan')
read_json(plan)
native = read_json(BASE + '/context-qsa-run4/analysis/RESULT.json')
audit = read_json(BASE + '/review/context-qsa-run4-postprocess-20261003/RAW-AUDIT.json')
actual = audit.fetch('actual')
require_check(audit['checker_exit'] == 0 && actual['scope'] == 'full' && actual['evidence_errors'].empty?, 'raw audit not complete')
require_check([actual['GROUP_closed'],actual['HTTP_verified'],actual['outputs_verified']] == [114,798,112488], 'raw totals differ')
require_check(native['diagnostic_exit'] == 3, 'preserve actual native exit')
restore = read_json(BASE + '/review/context-qsa-run4-restoration-20261002/PARENT-RESTORATION-REVIEW.json')
require_check(restore['gates'].length == 17 && restore['gates'].values.all?{|v|v == true}, 'restoration gates')
require_check([restore['requests'],restore['fast'],restore['slow']] == [53,24,0], 'restoration totals')
memory = read_json(BASE + '/review/context-qsa-run4-memory-review-20261002/actual-final-closed/MEMORY-REVIEW.json')
resource = {}
groups = {}
elapsed = {}
%w[A0 B A2].each do |arm|
  result = read_json(BASE + '/context-qsa-run4/control/attempt1/' + arm + '/http/RESULT.json')
  require_check([result['completed_groups'],result['HTTP_requests'],result['outputs']] == [38,266,37496], 'arm totals')
  elapsed[arm] = result['elapsed_s']
  result.fetch('group_records').each do |ref|
    path = local_path(ref.fetch('path'))
    group = read_json(path)
    require_check(PINS[path.sub(BASE + '/', '')] == ref.fetch('sha256'), 'group SHA differs')
    key = [arm, group.fetch('row_id'), group.fetch('ordinal')]
    require_check(!groups.key?(key), 'duplicate group')
    tokens = group.fetch('request_results').map do |request|
      raw = tokens_from_raw(request.fetch('raw_response'))
      require_check(raw == request.fetch('output_token_ids'), 'GROUP tokens differ from raw')
      raw
    end
    groups[key] = {group:group,tokens:tokens,path:path.sub(BASE + '/', '')}
  end
  mem = memory.fetch('arms').fetch(arm)
  client = mem.fetch('client_resource')
  require_check(client.fetch('checks').values.all?{|v|v == true}, 'client resource gate')
  require_check(mem['arm_failures'].empty? && mem['group_failures'].empty? && mem['first_recorded_resource_issue'].nil?, 'resource issue')
  resource[arm] = {client_peak_bytes:client['actual_cgroup_peak_bytes'],client_checks:client['checks'],sampled_max:mem['sampled_max_MiB_by_UUID']}
end

counts = Hash.new(0)
details = []
rows = []
group_status_counts = Hash.new(0)
native.fetch('rows').each do |row|
  pairs = []
  row_counts = Hash.new(0)
  row.fetch('token_equality').each{|v|group_status_counts[v['status']] += 1}
  2.times do |repeat|
    triple = %w[A0 B A2].map{|arm|groups.fetch([arm,row.fetch('row_id'),repeat])}
    metrics = triple.map{|g|g[:group].fetch('metrics').fetch('pooled_complete_interval_ms_per_output_token')}
    require_check(metrics == row.fetch('common_decode').fetch('paired_values')[repeat], 'native GROUP metric differs')
    independent_row = actual.fetch('rows').find{|r|r['row_id'] == row['row_id']}
    independent = independent_row.fetch('common_decode').fetch('paired_values')[repeat]
    require_check(metrics.zip(independent).all?{|x,y| x.nil? ? y.nil? : !y.nil? && (x-y).abs < 1e-10}, 'raw metric differs')
    pair = {repeat:repeat,ms_per_token:metrics}
    if metrics.none?(&:nil?)
      a0,b,a2 = metrics
      average_gain = 100*(1-b/((a0+a2)/2.0))
      drift = 100*(a0-a2).abs/((a0+a2)/2.0)
      require_check((average_gain-row['common_decode']['improvement_pct_per_pair'][repeat]).abs < 1e-10, 'gain differs')
      require_check((drift-row['common_decode']['AA_drift_pct_per_pair'][repeat]).abs < 1e-10, 'drift differs')
      pair.merge!(B_vs_A0_pct:100*(1-b/a0),B_vs_A2_pct:100*(1-b/a2),B_vs_mean_A_pct:average_gain,AA_drift_pct:drift)
    end
    pairs << pair
    triple[0][:tokens].length.times do |index|
      a0,b,a2 = triple.map{|g|g[:tokens][index]}
      kind = a0 == a2 ? (b == a0 ? 'ALL_THREE_EQUAL' : 'A0_A2_EQUAL_B_DIFFERS') : 'AA_SELF_VARIATION'
      counts[kind] += 1
      row_counts[kind] += 1
      next if kind == 'ALL_THREE_EQUAL'
      diff = {}
      [['A0_A2',a0,a2],['A0_B',a0,b],['A2_B',a2,b]].each do |name,x,y|
        next if x == y
        n = x.zip(y).index{|u,v|u != v}
        diff[name] = {first_token_index:n,left_token:x[n],right_token:y[n]}
      end
      details << {row_id:row['row_id'],repeat:repeat,request_index:index,classification:kind,first_differences:diff,groups:triple.map{|g|g[:path]}}
    end
  end
  rows << {row_id:row['row_id'],input_tokens:row['input_tokens'],concurrency:row['concurrency'],request_triple_counts:row_counts,pairs:pairs}
end
require_check(counts.values.sum == 146, 'main request triple count')
report = {status:'COMPLETE_EVIDENCE_WITH_REQUEST_LEVEL_OUTPUT_DISAGREEMENTS_NO_ADMISSION',plan_sha256:PLAN_SHA,
  method:'Independent Ruby raw-SSE main token extraction, GROUP and raw SHA verification; metrics compared against independent raw auditor and recomputed paired arithmetic. Original frozen files and exits unchanged.',
  raw_main_streams:438,main_tokens:112128,request_triples:146,request_triple_counts:counts,group_level_status_counts:group_status_counts,
  elapsed_HTTP_s:elapsed,rows:rows,output_disagreements:details,resources:resource,
  restoration:{pid:restore['pid'],invocation_id:restore['invocation_id'],requests:53,gates:17,fast:24,slow:0},
  limitations:['A0=A2 in one request is not a deterministic execution-plan or QSA causal proof.','Group-level AA_SELF_VARIATION can coexist with individual A0=A2/B-different requests.','n=2 is descriptive; SSE is not GPU timing or a hardware lower bound.','GPU maxima are samples, not continuous peaks.'],
  pins:PINS,source_sha256:Digest::SHA256.file(__FILE__).hexdigest}
File.open(File.join(__dir__,'ROOT-FINAL-REVIEW.json'),File::WRONLY|File::CREAT|File::EXCL,0600){|f|f.write(JSON.pretty_generate(report)+"\n")}
puts JSON.generate(status:report[:status],counts:counts,group_counts:group_status_counts,verified_pins:PINS.length)
