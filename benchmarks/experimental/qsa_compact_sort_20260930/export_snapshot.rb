require 'json'
require 'digest'
require 'fileutils'
require 'set'

SOURCE = '/private/tmp/flash-next-step58-20260930'
REMOTE = '/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930'
DEST = ARGV.fetch(0, '/private/tmp/qsa-git-export-20261003-v3')
raise 'export already exists' if File.exist?(DEST)
FileUtils.mkdir_p(DEST)
selected = Set.new
source_ext = /\.(py|cu|cpp|h|hpp|cuh|inc|sh|rb)$/
excluded = %r{/(evidence|build|__pycache__|archive1|admission|groups|sources|cpu-fixture|cpu-core|tested-source)/}
roots = %w[implementation w1-implementation service-implementation-retry2 context-qsa-prep1 context-qsa-control-prep4 context-qsa-transport-prep1]
roots.each do |root|
  Dir.glob(SOURCE + '/' + root + '/**/*').each do |path|
    relative = path.delete_prefix(SOURCE + '/')
    next unless File.file?(path) && !File.symlink?(path)
    next if ('/' + relative).match?(excluded)
    selected << relative if relative.match?(source_ext)
  end
  Dir.glob(SOURCE + '/' + root + '/*').each do |path|
    next unless File.file?(path) && path.match?(/\.(json|md)$/)
    selected << path.delete_prefix(SOURCE + '/')
  end
end
plan = JSON.parse(File.read(SOURCE + '/context-qsa-run4/control/plan.json'))
plan['files'].each_key do |path|
  next unless path.start_with?(REMOTE + '/')
  relative = path.delete_prefix(REMOTE + '/')
  next if ('/' + relative).match?(excluded)
  selected << relative if relative.match?(source_ext)
end
%w[
  implementation/reference54/source/IMPLEMENTATION.md
  implementation/reference54/source/ORIGINAL-LICENSE
  implementation/reference54/source/reference/release-manifest.json
  service-implementation-retry2/source/prod-w48-release.json
  context-qsa-control-prep4/source/restoration-external-reference-pins.json
  context-qsa-control-prep4/source/ASSEMBLY.json
  context-qsa-control-prep4/source/INTEGRATION-REVIEW.json
  context-qsa-prep1/sources/input-8192.json
  context-qsa-prep1/sources/input-32768.json
  review/context-qsa-independent-20261002/independent_review.py
  review/context-qsa-independent-20261002/partial_raw_review.py
  review/context-qsa-independent-20261002/ALGORITHM-REVIEW-GUIDE.md
  review/context-qsa-closure-20261002/REVIEW-PACK.md
  review/context-qsa-closure-20261002/PRE-CLOSURE-REVIEW-PACK.md
  review/context-qsa-closure-20261002/FINAL-LIVE-HEALTH.json
  review/context-qsa-closure-20261002/ROOT-LAUNCH-RECEIPT.json
  REVIEW-HANDOFF-20261003.md
  REVIEW-PROMPT-20261003.md
  implementation/reference54/CONTRACT.md
  context-qsa-prep1/matrix.frozen.json
  context-qsa-control-prep2/MANIFEST.json
  context-qsa-control-prep3/MANIFEST.json
  context-qsa-resource-control-prep1/CONTRACT.json
  service-w2-control-prep4/contract.json
  service-w2-control-prep4/manifest.json
  service-w2-control-prep4/command-contract.json
  service-w2-control-prep4/source/restoration-external-reference-pins.json
  service-w2-prep/manifest.json
  service-w2-prep/contract.json
  context-qsa-run4/control/plan.json
  context-qsa-run4/analysis/RESULT.json
  review/context-qsa-closure-20261002/root_review.rb
  review/context-qsa-closure-20261002/render_results.py
  review/context-qsa-closure-20261002/ROOT-FINAL-REVIEW.json
  review/context-qsa-closure-20261002/CONTEXT-RESULTS.md
  review/context-qsa-closure-20261002/ARCHIVE-READBACK.json
  review/context-qsa-run4-independent-20261002/audit_run4.py
  review/context-qsa-run4-memory-review-20261002/memory_review.py
  review/context-qsa-run4-restoration-20261002/collect_final_original.py
  review/context-qsa-run4-restoration-20261002/recompute_restoration.rb
  review/context-qsa-run4-restoration-20261002/PARENT-RESTORATION-REVIEW.json
  review/context-qsa-run4-postprocess-20261003/per_request_cross_review.py
  review/context-qsa-run4-postprocess-20261003/PER-REQUEST-CROSS-REVIEW.json
  review/context-qsa-run4-postprocess-20261003/CROSS-REVIEW.md
  review/context-qsa-run4-postprocess-20261003/POSTPROCESS-RESULT.json
].each do |path|
  raise 'missing required source: ' + path unless File.file?(SOURCE + '/' + path)
  selected << path
end
entries = selected.sort.map do |relative|
  path = File.join(SOURCE,relative)
  raise 'missing or symlink source: ' + relative unless File.file?(path) && !File.symlink?(path)
  bytes = File.binread(path)
  raise 'binary source: ' + relative if bytes.include?("\x00")
  raise 'oversized source: ' + relative if bytes.bytesize > 2 * 1024 * 1024
  raise 'credential signature: ' + relative if bytes.match?(/-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{16,}|(?:AKIA|ASIA)[A-Z0-9]{16}/)
  target = File.join(DEST,'snapshot',relative)
  FileUtils.mkdir_p(File.dirname(target))
  File.binwrite(target,bytes)
  {path:'snapshot/'+relative,host_path:REMOTE+'/'+relative,bytes:bytes.bytesize,sha256:Digest::SHA256.hexdigest(bytes)}
end
manifest = {
  purpose:'Source and bounded result snapshot for independent review; not runtime integration or a standalone replay distribution',
  host:'oh-my-gpu',host_evidence_root:REMOTE,
  tested_vllm_head:'2dd437062a971a83d1f905249d288269c8daf501',
  git_snapshot_base:'fe67339ddf862df0441a4e844fc664d9cafdffd0',
  plan_sha256:'355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4',
  host_archive_manifest_sha256:'e5e6824e1cecc2c938859df35e204dfb98d9332e2b691170d18dc90cc6e1e24e',
  copied_bytes_unchanged:true,files:entries,
  omitted:['compiled SO and build products','raw SSE/jsonl, logs, generated prompt groups and tensor fixtures (two seed JSON files included)','model weights and Python environments','provider credentials and isolated CLI state','bulk historical runtime evidence'],
  replay_limit:'Original contracts and absolute paths remain unchanged. Their full dependency manifests include omitted host artifacts; execution requires the original host archive and runtime. Do not bypass these guards.'
}
File.write(File.join(DEST,'SOURCE-MANIFEST.json'),JSON.pretty_generate(manifest)+"\n")
FileUtils.cp(__FILE__,File.join(DEST,'export_snapshot.rb'))
json_paths = entries.map { |e| e[:path] }.select { |p| p.end_with?('.json') }
json_paths += %w[SOURCE-MANIFEST.json HOST-EVIDENCE-INDEX.json PUBLICATION-CHECKS.json]
ignore = "# Explicitly track the reviewed JSON snapshot; the parent excludes benchmark JSON.\n"
ignore += json_paths.sort.map { |p| '!/' + p }.join("\n") + "\n"
ignore += "__pycache__/\n*.pyc\n*.so\n*.pt\n*.bin\n*.jsonl\n"
File.write(File.join(DEST,'.gitignore'),ignore)
puts JSON.generate(files:entries.length,bytes:entries.sum{|x|x[:bytes]},max_file:entries.max_by{|x|x[:bytes]}.slice(:path,:bytes),export:DEST)
