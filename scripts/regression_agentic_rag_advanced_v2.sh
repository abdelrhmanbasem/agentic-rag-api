#!/usr/bin/env bash
set -Eeuo pipefail

# Advanced Agentic RAG regression suite v2.
# Uses dynamic available slots from API debug state instead of assuming 3 PM / 5 PM are still available.
# Includes v6.45 code-expert cost/runtime control validation.
#
# Usage:
#   export API_URL="http://localhost:8010"
#   export API_KEY="YOUR_API_KEY"
#   bash scripts/regression_agentic_rag_advanced_v2.sh

API_URL="${API_URL:-http://localhost:8010}"
API_KEY="${API_KEY:-${APP_SECRET:-}}"
ASSISTANT_ID="${ASSISTANT_ID:-service_center_agentic_rag}"
USER_ID="${USER_ID:-201554354929@s.whatsapp.net}"
OUT_ROOT="${OUT_ROOT:-/root}"
TEST_DATE="${TEST_DATE:-2026-06-07}"

if [ -z "$API_KEY" ]; then
  echo "FAIL: API_KEY or APP_SECRET must be exported before running this script."
  exit 1
fi

pass_count=0
fail_count=0

pass(){ pass_count=$((pass_count+1)); echo "PASS: $*"; }
fail(){ fail_count=$((fail_count+1)); echo "FAIL: $*"; }
log(){ printf '\n%s\n' "$*"; }

json_ok(){
  local file="$1"
  if python3 -m json.tool "$file" >/dev/null 2>&1; then
    pass "valid JSON: $file"
  else
    fail "invalid JSON: $file"
    cat "$file" || true
    return 1
  fi
}

container_id(){ docker ps --filter "name=rag-api" -q | head -n 1; }

clear_conversation(){
  local cid="$1"
  local out_dir="$2"

  curl -s -X POST "$API_URL/conversations/$ASSISTANT_ID/$cid/clear"     -H "x-api-key: $API_KEY" > "$out_dir/clear.json"
  json_ok "$out_dir/clear.json"

  python3 - "$out_dir/clear.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
assert d.get("cleared") is True, d
print("clear ok")
PY
}

send_msg(){
  local conversation_id="$1"
  local mid="$2"
  local msg="$3"
  local out="$4"
  local key="${conversation_id}_${mid}"

  curl -s     -H "Content-Type: application/json"     -H "x-api-key: $API_KEY"     -X POST "$API_URL/chat"     -d "{
      \"assistant_id\": \"$ASSISTANT_ID\",
      \"user_id\": \"$USER_ID\",
      \"conversation_id\": \"$conversation_id\",
      \"message\": \"$msg\",
      \"channel\": \"advanced_regression\",
      \"message_id\": \"$key\",
      \"idempotency_key\": \"$key\",
      \"debug\": true
    }" > "$out"

  json_ok "$out"
}

slot_count(){
  python3 - "$1" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
s=(d.get("debug") or {}).get("state_after") or {}
print(len(s.get("available_slots") or []))
PY
}

slot_text(){
  python3 - "$1" "$2" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
idx=int(sys.argv[2])
s=(d.get("debug") or {}).get("state_after") or {}
slot=(s.get("available_slots") or [])[idx]
print(slot.get("time_text") or slot.get("time") or "")
PY
}

slot_time(){
  python3 - "$1" "$2" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
idx=int(sys.argv[2])
s=(d.get("debug") or {}).get("state_after") or {}
slot=(s.get("available_slots") or [])[idx]
print(slot.get("time") or "")
PY
}

validate_deployment(){
  log "=== 1) Architecture and hardcoding validation ==="
  local cid
  cid="$(container_id || true)"
  if [ -z "$cid" ]; then fail "rag-api Docker container found"; return 1; fi
  pass "rag-api Docker container found: $cid"

  docker exec -i "$cid" sh -lc '
set -eu
python3 -m py_compile app/graph.py
python3 -m py_compile app/subagents/booking_subagent.py
python3 -m py_compile app/main.py
python3 -m py_compile app/config.py
python3 -m json.tool /app/configs/service_center_agentic_rag/domain_bundle.json >/dev/null
grep -q "6.36-manifest-history-limit-no-hardcoding-graph" app/graph.py
grep -q "6.45-code-expert-cost-smartness-no-hardcoding-graph" app/graph.py
grep -q "semantic_extraction_node" app/graph.py
grep -q "pre_response_guardrail_node" app/graph.py
grep -q "graph_extract_pending_required_details_from_patterns" app/graph.py
grep -q "WHAT IS ALREADY KNOWN" app/graph.py
grep -q "ONE QUESTION" app/graph.py
grep -q "MANIFEST_HISTORY_LIMIT" app/graph.py
grep -q "compact_variables_for_response" app/graph.py
grep -q "get_response_model" app/graph.py
grep -q "should_retry_full_manifest_for_stripped_updates" app/graph.py
grep -q "should_skip_subagent_reasoning_after_tool" app/graph.py
grep -q "build_response_guidance_block" app/graph.py
grep -q "SUBAGENT_REASONING_MAX_TOKENS" app/graph.py
! grep -q "max_tokens=700" app/graph.py
! grep -q "messages\[-12:\]" app/graph.py
grep -q "6.30-skip-early-slot-guard-in-detail-stage-no-hardcoding" app/subagents/booking_subagent.py
grep -q "stage not in {awaiting_confirmation_stage, awaiting_customer_details_stage}" app/subagents/booking_subagent.py
grep -q "mirror_canonical_customer_profile_to_booking_profile" app/subagents/booking_subagent.py
grep -q "6.33-config-driven-main-error-handling-no-hardcoding" app/main.py
grep -q "6.34-runtime-controls-no-hardcoding" app/config.py
grep -q "6.45-code-expert-runtime-controls-no-hardcoding" app/config.py
grep -q "MODEL_RESPONSE_SIMPLE" app/config.py
grep -q "MAX_SUBAGENT_REASONING_TOKENS" app/config.py
grep -q "RESPONSE_MODEL_ROUTING_GLOBAL_ENABLED" app/config.py
grep -q "6.31-semantic-variable-extraction-config-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.45-code-expert-cost-smartness-config-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"semantic_variable_extraction\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"manifest_retry_policy\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"response_model_routing\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"response_compaction\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"quality_guard_policy\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"subagent_reasoning_policy\"" /app/configs/service_center_agentic_rag/domain_bundle.json
'
  pass "architecture markers and hardcoding checks"
}

validate_health(){
  log "=== 2) API health/config validation ==="
  local out_dir="$OUT_ROOT/agentic_advanced_health"
  mkdir -p "$out_dir"

  curl -s "$API_URL/health" > "$out_dir/health.json"
  json_ok "$out_dir/health.json"
  python3 - "$out_dir/health.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
print("health ok")
PY
  pass "health ok"

  curl -s -H "x-api-key: $API_KEY" "$API_URL/config-source/$ASSISTANT_ID" > "$out_dir/config_source.json"
  json_ok "$out_dir/config_source.json"
  python3 - "$out_dir/config_source.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("source") == "domain_bundle", d
assert d.get("assistant_found") is True, d
assert d.get("schema_found") is True, d
print("config source ok")
PY
  pass "config-source ok"
}

scenario_dynamic_baseline(){
  log "=== 3) Scenario A: dynamic available-slot booking ==="

  local conversation_id="advanced_dynamic_baseline_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_advanced_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir"

  send_msg "$conversation_id" "001" "انا ساكن في العبور قولي اقرب فرع واولي المواعيد المتاحة يوم $TEST_DATE" "$out_dir/turn_001.json"

  local count
  count="$(slot_count "$out_dir/turn_001.json")"
  if [ "$count" -lt 1 ]; then
    fail "Scenario A requires at least one available slot on $TEST_DATE"
    python3 - "$out_dir/turn_001.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"answer": d.get("answer"), "state": (d.get("debug") or {}).get("state_after") or {}}, ensure_ascii=False, indent=2))
PY
    return 1
  fi

  local chosen_index=0
  local chosen_text
  local chosen_time
  chosen_text="$(slot_text "$out_dir/turn_001.json" "$chosen_index")"
  chosen_time="$(slot_time "$out_dir/turn_001.json" "$chosen_index")"
  echo "Scenario A selected slot: $chosen_text ($chosen_time)"

  send_msg "$conversation_id" "002" "معاد الساعة $chosen_text هيكون مناسب معايا" "$out_dir/turn_002.json"
  send_msg "$conversation_id" "003" "اه" "$out_dir/turn_003.json"
  send_msg "$conversation_id" "004" "اسمي عبدالرحمن باسم" "$out_dir/turn_004.json"
  send_msg "$conversation_id" "005" "رقم العربيه ب ج د ٥٥٥" "$out_dir/turn_005.json"

  python3 - "$out_dir" "$chosen_time" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
expected=sys.argv[2]
def load(n): return json.load(open(out/n, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def booking(t): return state(t).get("booking") or {}
def profile(t): return state(t).get("customer_profile") or {}
t1,t2,t3,t4,t5=[load(f"turn_00{i}.json") for i in range(1,6)]
errors=[]
s1=state(t1)
if s1.get("selected_branch")!="Nasr City": errors.append("T1 branch not Nasr City")
if s1.get("appointment_date")!="2026-06-07": errors.append("T1 date not preserved")
if len(s1.get("available_slots") or []) < 1: errors.append("T1 expected at least one slot")
s2=state(t2); b2=booking(t2); p2=b2.get("pending") or {}
if s2.get("appointment_time")!=expected: errors.append(f"T2 time expected {expected} got {s2.get('appointment_time')}")
if p2.get("time")!=expected: errors.append(f"T2 pending time expected {expected} got {p2.get('time')}")
if b2.get("stage")!="awaiting_confirmation": errors.append(f"T2 stage expected awaiting_confirmation got {b2.get('stage')}")
s3=state(t3); b3=booking(t3)
if s3.get("customer_confirmed_booking") is not True: errors.append("T3 confirmation not true")
if b3.get("stage")!="awaiting_customer_details": errors.append("T3 stage not awaiting_customer_details")
s4=state(t4); p4=profile(t4); bp4=booking(t4).get("customer_profile") or {}
if p4.get("full_name")!="عبدالرحمن باسم": errors.append("T4 name not saved")
if bp4.get("full_name")!="عبدالرحمن باسم": errors.append("T4 booking profile name not mirrored")
s5=state(t5); b5=booking(t5); p5=profile(t5); bp5=b5.get("customer_profile") or {}
answer=t5.get("answer") or ""
if b5.get("stage")!="confirmed": errors.append(f"T5 stage expected confirmed got {b5.get('stage')}")
if s5.get("appointment_time")!=expected: errors.append(f"T5 slot expected {expected} got {s5.get('appointment_time')}")
if p5.get("plate_number")!="ب ج د 555": errors.append(f"T5 plate wrong: {p5.get('plate_number')}")
if bp5.get("plate_number")!="ب ج د 555": errors.append("T5 booking profile plate wrong")
if "VIS-" not in answer: errors.append("T5 no visit ID in answer")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print(json.dumps({"scenario":"dynamic_baseline","stage":b5.get("stage"),"time":s5.get("appointment_time"),"answer":answer}, ensure_ascii=False, indent=2))
PY
  pass "Scenario A dynamic baseline booking"
}

scenario_hesitation_change_mind_dynamic(){
  log "=== 4) Scenario B: hesitation + change mind with dynamic slots ==="

  local conversation_id="advanced_change_mind_dynamic_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_advanced_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir"

  send_msg "$conversation_id" "001" "انا ساكن في العبور بس مش متأكد، ممكن تقولي اقرب فرع والمواعيد المتاحة يوم $TEST_DATE؟" "$out_dir/turn_001.json"

  local count
  count="$(slot_count "$out_dir/turn_001.json")"
  if [ "$count" -lt 2 ]; then
    fail "Scenario B requires at least two available slots on $TEST_DATE"
    python3 - "$out_dir/turn_001.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
s=(d.get("debug") or {}).get("state_after") or {}
print(json.dumps({"answer": d.get("answer"), "available_slots": s.get("available_slots") or []}, ensure_ascii=False, indent=2))
PY
    return 1
  fi

  local first_index=0
  local second_index=$((count - 1))
  local first_text first_time second_text second_time
  first_text="$(slot_text "$out_dir/turn_001.json" "$first_index")"
  first_time="$(slot_time "$out_dir/turn_001.json" "$first_index")"
  second_text="$(slot_text "$out_dir/turn_001.json" "$second_index")"
  second_time="$(slot_time "$out_dir/turn_001.json" "$second_index")"

  echo "Scenario B first slot: $first_text ($first_time)"
  echo "Scenario B changed slot: $second_text ($second_time)"

  send_msg "$conversation_id" "002" "مش عارف، ممكن تفكرني بالمواعيد تاني؟" "$out_dir/turn_002.json"
  send_msg "$conversation_id" "003" "خليها الساعة $first_text" "$out_dir/turn_003.json"
  send_msg "$conversation_id" "004" "لا استنى، خليها الساعة $second_text بدل المعاد الأول" "$out_dir/turn_004.json"
  send_msg "$conversation_id" "005" "اه كمل" "$out_dir/turn_005.json"
  send_msg "$conversation_id" "006" "اسمي عبدالرحمن باسم" "$out_dir/turn_006.json"
  send_msg "$conversation_id" "007" "رقم العربيه ب ج د ٥٥٥" "$out_dir/turn_007.json"

  python3 - "$out_dir" "$first_time" "$second_time" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
first_expected=sys.argv[2]
second_expected=sys.argv[3]
def load(n): return json.load(open(out/n, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def booking(t): return state(t).get("booking") or {}
def profile(t): return state(t).get("customer_profile") or {}
turns=[load(f"turn_00{i}.json") for i in range(1,8)]
errors=[]
s2=state(turns[1])
if s2.get("selected_branch")!="Nasr City": errors.append("T2 branch lost during hesitation")
if s2.get("appointment_date")!="2026-06-07": errors.append("T2 date lost during hesitation")
if len(s2.get("available_slots") or []) < 1: errors.append("T2 slots lost during hesitation")
s3=state(turns[2]); p3=(booking(turns[2]).get("pending") or {})
if s3.get("appointment_time")!=first_expected: errors.append(f"T3 expected {first_expected} got {s3.get('appointment_time')}")
if p3.get("time")!=first_expected: errors.append(f"T3 pending expected {first_expected} got {p3.get('time')}")
s4=state(turns[3]); p4=(booking(turns[3]).get("pending") or {})
if s4.get("appointment_time")!=second_expected: errors.append(f"T4 expected changed time {second_expected} got {s4.get('appointment_time')}")
if p4.get("time")!=second_expected: errors.append(f"T4 pending expected {second_expected} got {p4.get('time')}")
s5=state(turns[4]); b5=booking(turns[4])
if s5.get("customer_confirmed_booking") is not True: errors.append("T5 confirmation not true")
if s5.get("appointment_time")!=second_expected: errors.append("T5 changed slot not preserved")
if b5.get("stage")!="awaiting_customer_details": errors.append(f"T5 stage expected awaiting_customer_details got {b5.get('stage')}")
s7=state(turns[6]); b7=booking(turns[6]); p7=profile(turns[6]); bp7=b7.get("customer_profile") or {}
answer=turns[6].get("answer") or ""
if b7.get("stage")!="confirmed": errors.append(f"T7 expected confirmed got {b7.get('stage')}")
if s7.get("appointment_time")!=second_expected: errors.append(f"T7 expected final time {second_expected} got {s7.get('appointment_time')}")
if p7.get("full_name")!="عبدالرحمن باسم": errors.append("T7 full_name wrong")
if p7.get("plate_number")!="ب ج د 555": errors.append("T7 plate wrong")
if bp7.get("plate_number")!="ب ج د 555": errors.append("T7 booking profile plate wrong")
if "VIS-" not in answer: errors.append("T7 no visit ID in answer")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print(json.dumps({"scenario":"hesitation_change_mind_dynamic","stage":b7.get("stage"),"first_time":first_expected,"final_time":s7.get("appointment_time"),"answer":answer}, ensure_ascii=False, indent=2))
PY
  pass "Scenario B hesitation + changed slot booking"
}

scenario_date_change(){
  log "=== 5) Scenario C: date change clears stale slot ==="
  local conversation_id="advanced_date_change_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_advanced_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir"
  send_msg "$conversation_id" "001" "انا ساكن في العبور قولي المواعيد المتاحة يوم 2026-06-07" "$out_dir/turn_001.json"
  send_msg "$conversation_id" "002" "لا مش اليوم ده، شوف يوم 2026-06-08" "$out_dir/turn_002.json"
  python3 - "$out_dir" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
def load(n): return json.load(open(out/n, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def booking(t): return state(t).get("booking") or {}
t1=load("turn_001.json"); t2=load("turn_002.json")
s1=state(t1); s2=state(t2); b2=booking(t2)
errors=[]
if s1.get("appointment_date")!="2026-06-07": errors.append("T1 initial date wrong")
if s2.get("appointment_date")!="2026-06-08": errors.append(f"T2 date change expected 2026-06-08 got {s2.get('appointment_date')}")
if s2.get("appointment_time") not in [None, ""]: errors.append(f"T2 appointment_time should clear after date change, got {s2.get('appointment_time')}")
if (b2.get("pending") or {}).get("time"): errors.append("T2 booking.pending.time should clear after date change")
if errors:
    print("\n".join(errors)); raise SystemExit(1)
print(json.dumps({"scenario":"date_change","old_date":s1.get("appointment_date"),"new_date":s2.get("appointment_date"),"stage":b2.get("stage"),"slots_found":s2.get("slots_found")}, ensure_ascii=False, indent=2))
PY
  pass "Scenario C date change does not keep stale slot"
}

scenario_semantic_config(){
  log "=== 6) Scenario D: semantic extraction config sanity ==="
  local cid
  cid="$(container_id || true)"
  if [ -z "$cid" ]; then fail "semantic config: no Docker container"; return 1; fi
  docker exec -i "$cid" python3 - <<'PY'
import json
bundle=json.load(open("/app/configs/service_center_agentic_rag/domain_bundle.json", encoding="utf-8"))
assistant=bundle.get("assistant") or {}
sem=assistant.get("semantic_variable_extraction") or {}
assert sem.get("enabled") is True, sem
fields=sem.get("fields") or []
ids={f.get("id") for f in fields if isinstance(f, dict)}
required={"customer_full_name","customer_phone","customer_plate_number"}
missing=required-ids
assert not missing, missing
for field in fields:
    if isinstance(field, dict) and field.get("id") in required:
        for key in ["target_path","description","output_format","validation_description"]:
            assert field.get(key), (field.get("id"), key)
answer_safety=assistant.get("answer_safety") or {}
assert answer_safety.get("record_id_label"), answer_safety
assert answer_safety.get("record_id_format"), answer_safety
fallback=assistant.get("fallback_messages") or {}
assert fallback.get("graph_error"), fallback
assert fallback.get("default_final"), fallback

manifest_ctx=assistant.get("manifest_context") or {}
assert int(manifest_ctx.get("previous_manifest_summary_max_chars") or 0) <= 600, manifest_ctx
retry=assistant.get("manifest_retry_policy") or {}
assert float(retry.get("source_of_truth_strip_ratio_threshold") or 0) >= 0.6, retry
assert int(retry.get("min_stripped_updates_for_retry") or 0) >= 3, retry
routing=assistant.get("response_model_routing") or {}
assert routing.get("enabled") is True and routing.get("simple_model") and routing.get("default_model"), routing
compaction=assistant.get("response_compaction") or {}
assert compaction.get("enabled") is True and compaction.get("exclude_variable_paths_when_tool_result_has"), compaction
quality=assistant.get("quality_guard_policy") or {}
assert quality.get("enabled", True) is not False and int(quality.get("long_answer_chars") or 0) >= 200, quality
subagent_reasoning=assistant.get("subagent_reasoning_policy") or {}
assert subagent_reasoning.get("enabled", True) is not False and subagent_reasoning.get("skip_on_clean_executor_result") is True, subagent_reasoning
print("semantic/no-hardcoding/code-expert config ok")
PY
  pass "Scenario D semantic config sanity"
}

summary(){
  log "=== Advanced regression v2 summary ==="
  echo "Passes: $pass_count"
  echo "Failures: $fail_count"
  if [ "$fail_count" -ne 0 ]; then
    echo ""
    echo "ADVANCED REGRESSION V2 FAILED"
    echo "Inspect /root/agentic_advanced_* directories for debug JSON."
    exit 1
  fi
  echo ""
  echo "ALL ADVANCED REGRESSION V2 CHECKS PASSED"
}

validate_deployment
validate_health
scenario_dynamic_baseline
scenario_hesitation_change_mind_dynamic
scenario_date_change
scenario_semantic_config
summary
