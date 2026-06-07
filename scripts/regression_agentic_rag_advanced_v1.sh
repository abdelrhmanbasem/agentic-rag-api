#!/usr/bin/env bash
set -Eeuo pipefail

# Advanced Agentic RAG regression suite.
#
# Covers:
# - deployed architecture markers
# - health/config-source
# - baseline 5-turn confirmed booking
# - hesitation / relisting slots without losing state
# - user changing selected slot before confirmation
# - final create_booking with changed slot
# - idempotency duplicate replay
# - persisted conversation state
# - semantic extraction config sanity
#
# Usage:
#   export API_URL="http://localhost:8010"
#   export API_KEY="YOUR_API_KEY"
#   bash scripts/regression_agentic_rag_advanced_v1.sh

API_URL="${API_URL:-http://localhost:8010}"
API_KEY="${API_KEY:-${APP_SECRET:-}}"
ASSISTANT_ID="${ASSISTANT_ID:-service_center_agentic_rag}"
USER_ID="${USER_ID:-201554354929@s.whatsapp.net}"
OUT_ROOT="${OUT_ROOT:-/root}"

if [ -z "$API_KEY" ]; then
  echo "FAIL: API_KEY or APP_SECRET must be exported before running this script."
  exit 1
fi

pass_count=0
fail_count=0

pass() {
  pass_count=$((pass_count + 1))
  echo "PASS: $*"
}

fail() {
  fail_count=$((fail_count + 1))
  echo "FAIL: $*"
}

log() {
  printf '\n%s\n' "$*"
}

json_ok() {
  local file="$1"
  if python3 -m json.tool "$file" >/dev/null 2>&1; then
    pass "valid JSON: $file"
  else
    fail "invalid JSON: $file"
    echo "Raw content:"
    cat "$file" || true
    return 1
  fi
}

container_id() {
  docker ps --filter "name=rag-api" -q | head -n 1
}

clear_conversation() {
  local cid="$1"
  local out_dir="$2"

  curl -s -X POST "$API_URL/conversations/$ASSISTANT_ID/$cid/clear" \
    -H "x-api-key: $API_KEY" > "$out_dir/clear.json"

  json_ok "$out_dir/clear.json"

  python3 - "$out_dir/clear.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
assert d.get("cleared") is True, d
print("clear ok")
PY

  local docker_cid
  docker_cid="$(container_id || true)"
  if [ -n "$docker_cid" ]; then
    docker exec -i "$docker_cid" sh -lc "rm -f '/app/data/conversations/$ASSISTANT_ID/$cid.json' || true" >/dev/null 2>&1 || true
  fi
}

send_msg() {
  local conversation_id="$1"
  local mid="$2"
  local msg="$3"
  local out="$4"
  local key="${conversation_id}_${mid}"

  curl -s \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -X POST "$API_URL/chat" \
    -d "{
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

validate_deployment() {
  log "=== 1) Architecture and hardcoding validation ==="

  local cid
  cid="$(container_id || true)"

  if [ -z "$cid" ]; then
    fail "rag-api Docker container found"
    return 1
  fi

  pass "rag-api Docker container found: $cid"

  local out_dir="$OUT_ROOT/agentic_advanced_deploy_validation"
  mkdir -p "$out_dir"

  docker exec -i "$cid" sh -lc '
set -eu

python3 -m py_compile app/graph.py
python3 -m py_compile app/subagents/booking_subagent.py
python3 -m py_compile app/main.py
python3 -m py_compile app/config.py
python3 -m json.tool /app/configs/service_center_agentic_rag/domain_bundle.json >/dev/null

grep -q "6.36-manifest-history-limit-no-hardcoding-graph" app/graph.py
grep -q "semantic_extraction_node" app/graph.py
grep -q "pre_response_guardrail_node" app/graph.py
grep -q "graph_extract_pending_required_details_from_patterns" app/graph.py
grep -q "WHAT IS ALREADY KNOWN" app/graph.py
grep -q "ONE QUESTION" app/graph.py
grep -q "MANIFEST_HISTORY_LIMIT" app/graph.py
! grep -q "messages\[-12:\]" app/graph.py

grep -q "6.30-skip-early-slot-guard-in-detail-stage-no-hardcoding" app/subagents/booking_subagent.py
grep -q "stage not in {awaiting_confirmation_stage, awaiting_customer_details_stage}" app/subagents/booking_subagent.py
grep -q "mirror_canonical_customer_profile_to_booking_profile" app/subagents/booking_subagent.py

grep -q "6.33-config-driven-main-error-handling-no-hardcoding" app/main.py
grep -q "api_error_messages" app/main.py
! grep -q "حصلت مشكلة، ممكن تحاول تاني" app/main.py

grep -q "6.34-runtime-controls-no-hardcoding" app/config.py
grep -q "MAX_EXTRACTION_TOKENS" app/config.py
grep -q "SEMANTIC_EXTRACTION_GLOBAL_ENABLED" app/config.py
grep -q "SIMPLE_RESPONSE_HISTORY_LIMIT" app/config.py

grep -q "6.31-semantic-variable-extraction-config-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"semantic_variable_extraction\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"customer_plate_number\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"record_id_label\"" /app/configs/service_center_agentic_rag/domain_bundle.json
' > "$out_dir/deploy_validation.txt" 2>&1 \
    && pass "architecture markers and hardcoding checks" \
    || {
      fail "architecture markers and hardcoding checks"
      cat "$out_dir/deploy_validation.txt"
      return 1
    }
}

validate_health() {
  log "=== 2) API health/config validation ==="

  local out_dir="$OUT_ROOT/agentic_advanced_health"
  mkdir -p "$out_dir"

  curl -s "$API_URL/health" > "$out_dir/health.json"
  json_ok "$out_dir/health.json"

  python3 - "$out_dir/health.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
assert d.get("service"), d
print("health ok")
PY
  pass "health ok"

  curl -s \
    -H "x-api-key: $API_KEY" \
    "$API_URL/config-source/$ASSISTANT_ID" > "$out_dir/config_source.json"

  json_ok "$out_dir/config_source.json"

  python3 - "$out_dir/config_source.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("source") == "domain_bundle", d
assert d.get("assistant_found") is True, d
assert d.get("schema_found") is True, d
assert "6.31" in str(d.get("architecture_version") or ""), d
print("config source ok")
PY
  pass "config-source ok"
}

scenario_baseline_booking() {
  log "=== 3) Scenario A: baseline 5-turn booking ==="

  local conversation_id="advanced_baseline_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_advanced_${conversation_id}"
  mkdir -p "$out_dir"

  clear_conversation "$conversation_id" "$out_dir"

  send_msg "$conversation_id" "001" "انا ساكن في العبور قولي اقرب فرع واولي المواعيد المتاحة يوم 2026-06-07" "$out_dir/turn_001.json"
  send_msg "$conversation_id" "002" "معاد الساعة 3 هيكون مناسب معايا" "$out_dir/turn_002.json"
  send_msg "$conversation_id" "003" "اه" "$out_dir/turn_003.json"
  send_msg "$conversation_id" "004" "اسمي عبدالرحمن باسم" "$out_dir/turn_004.json"
  send_msg "$conversation_id" "005" "رقم العربيه ب ج د ٥٥٥" "$out_dir/turn_005.json"

  python3 - "$out_dir" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])

def load(n):
    return json.load(open(out / n, encoding="utf-8"))

def state(t):
    return (t.get("debug") or {}).get("state_after") or {}

def booking(t):
    return state(t).get("booking") or {}

def profile(t):
    return state(t).get("customer_profile") or {}

t1,t2,t3,t4,t5=[load(f"turn_00{i}.json") for i in range(1,6)]

errors=[]

s1=state(t1)
if s1.get("selected_branch")!="Nasr City": errors.append("T1 branch not Nasr City")
if s1.get("appointment_date")!="2026-06-07": errors.append("T1 date not preserved")
if s1.get("slots_found") is not True: errors.append("T1 slots not found")
if len(s1.get("available_slots") or []) < 5: errors.append("T1 expected at least 5 slots")

s2=state(t2); b2=booking(t2); p2=b2.get("pending") or {}
if s2.get("appointment_time")!="15:00": errors.append("T2 time not 15:00")
if p2.get("time")!="15:00": errors.append("T2 pending time not 15:00")
if b2.get("stage")!="awaiting_confirmation": errors.append("T2 stage not awaiting_confirmation")

s3=state(t3); b3=booking(t3)
if s3.get("customer_confirmed_booking") is not True: errors.append("T3 confirmation not true")
if b3.get("stage")!="awaiting_customer_details": errors.append("T3 stage not awaiting_customer_details")

s4=state(t4); p4=profile(t4); bp4=booking(t4).get("customer_profile") or {}
if p4.get("full_name")!="عبدالرحمن باسم": errors.append("T4 name not saved")
if bp4.get("full_name")!="عبدالرحمن باسم": errors.append("T4 booking profile name not mirrored")
if s4.get("appointment_time")!="15:00": errors.append("T4 slot was lost")

s5=state(t5); b5=booking(t5); p5=profile(t5); bp5=b5.get("customer_profile") or {}
answer=t5.get("answer") or ""
if b5.get("stage")!="confirmed": errors.append("T5 stage not confirmed")
if s5.get("appointment_time")!="15:00": errors.append("T5 slot not preserved")
if p5.get("full_name")!="عبدالرحمن باسم": errors.append("T5 name wrong")
if p5.get("plate_number")!="ب ج د 555": errors.append("T5 plate wrong")
if bp5.get("plate_number")!="ب ج د 555": errors.append("T5 booking profile plate wrong")
if "VIS-" not in answer: errors.append("T5 no visit ID in answer")
for bad in ["هسألك عن مواعيد", "الحجز لسه مش مؤكد", "اختار معاد"]:
    if bad in answer:
        errors.append(f"T5 bad failure phrase present: {bad}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print(json.dumps({
  "scenario": "baseline",
  "stage": b5.get("stage"),
  "time": s5.get("appointment_time"),
  "full_name": p5.get("full_name"),
  "plate_number": p5.get("plate_number"),
  "answer": answer
}, ensure_ascii=False, indent=2))
PY
  pass "Scenario A baseline booking"
}

scenario_hesitation_and_change_mind() {
  log "=== 4) Scenario B: hesitation + relist + change selected slot ==="

  local conversation_id="advanced_hesitation_change_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_advanced_${conversation_id}"
  mkdir -p "$out_dir"

  clear_conversation "$conversation_id" "$out_dir"

  send_msg "$conversation_id" "001" "انا ساكن في العبور بس مش متأكد، ممكن تقولي اقرب فرع والمواعيد المتاحة يوم 2026-06-07؟" "$out_dir/turn_001.json"
  send_msg "$conversation_id" "002" "مش عارف، ممكن تفكرني بالمواعيد تاني؟" "$out_dir/turn_002.json"
  send_msg "$conversation_id" "003" "خليها الساعة 3" "$out_dir/turn_003.json"
  send_msg "$conversation_id" "004" "لا استنى، خليها الساعة 5 بدل 3" "$out_dir/turn_004.json"
  send_msg "$conversation_id" "005" "اه كمل" "$out_dir/turn_005.json"
  send_msg "$conversation_id" "006" "اسمي عبدالرحمن باسم" "$out_dir/turn_006.json"
  send_msg "$conversation_id" "007" "رقم العربيه ب ج د ٥٥٥" "$out_dir/turn_007.json"

  python3 - "$out_dir" <<'PY'
import json, sys
from pathlib import Path

out=Path(sys.argv[1])

def load(n): return json.load(open(out/n, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def booking(t): return state(t).get("booking") or {}
def profile(t): return state(t).get("customer_profile") or {}

turns=[load(f"turn_00{i}.json") for i in range(1,8)]
errors=[]

# Hesitation/relist should preserve available slots and not ask for branch/date again.
s2=state(turns[1])
if s2.get("selected_branch")!="Nasr City": errors.append("T2 branch lost during hesitation")
if s2.get("appointment_date")!="2026-06-07": errors.append("T2 date lost during hesitation")
if len(s2.get("available_slots") or []) < 5: errors.append("T2 slots lost during hesitation")

# Select 3 PM.
s3=state(turns[2]); p3=(booking(turns[2]).get("pending") or {})
if s3.get("appointment_time")!="15:00": errors.append(f"T3 expected 15:00 got {s3.get('appointment_time')}")
if p3.get("time")!="15:00": errors.append("T3 pending time not 15:00")

# Change mind to 5 PM before confirmation.
s4=state(turns[3]); p4=(booking(turns[3]).get("pending") or {})
if s4.get("appointment_time")!="17:00": errors.append(f"T4 expected changed time 17:00 got {s4.get('appointment_time')}")
if p4.get("time")!="17:00": errors.append(f"T4 pending time expected 17:00 got {p4.get('time')}")
if p4.get("section")!="Quick Service": errors.append(f"T4 section expected Quick Service got {p4.get('section')}")

# Confirm changed slot.
s5=state(turns[4]); b5=booking(turns[4])
if s5.get("customer_confirmed_booking") is not True: errors.append("T5 confirmation not true")
if s5.get("appointment_time")!="17:00": errors.append("T5 changed slot not preserved after confirmation")
if b5.get("stage")!="awaiting_customer_details": errors.append("T5 not awaiting customer details")

# Name and plate complete the booking.
s7=state(turns[6]); b7=booking(turns[6]); p7=profile(turns[6]); bp7=b7.get("customer_profile") or {}
answer=turns[6].get("answer") or ""
if b7.get("stage")!="confirmed": errors.append(f"T7 expected confirmed got {b7.get('stage')}")
if s7.get("appointment_time")!="17:00": errors.append(f"T7 expected final time 17:00 got {s7.get('appointment_time')}")
if p7.get("full_name")!="عبدالرحمن باسم": errors.append("T7 full_name wrong")
if p7.get("plate_number")!="ب ج د 555": errors.append("T7 plate wrong")
if bp7.get("full_name")!="عبدالرحمن باسم": errors.append("T7 booking profile name wrong")
if bp7.get("plate_number")!="ب ج د 555": errors.append("T7 booking profile plate wrong")
if "VIS-" not in answer: errors.append("T7 no visit ID in answer")
if "5:00 PM" not in answer and "17:00" not in answer and "٥" not in answer:
    errors.append(f"T7 answer does not mention changed 5 PM slot: {answer}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print(json.dumps({
  "scenario": "hesitation_change_mind",
  "stage": b7.get("stage"),
  "final_time": s7.get("appointment_time"),
  "section": (b7.get("pending") or {}).get("section"),
  "answer": answer
}, ensure_ascii=False, indent=2))
PY
  pass "Scenario B hesitation + changed slot booking"
}

scenario_date_change_after_slots() {
  log "=== 5) Scenario C: date change clears stale slot and relists ==="

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

t1=load("turn_001.json")
t2=load("turn_002.json")
s1=state(t1); s2=state(t2); b2=booking(t2)
errors=[]

if s1.get("appointment_date")!="2026-06-07": errors.append("T1 initial date wrong")
if s2.get("appointment_date")!="2026-06-08": errors.append(f"T2 date change expected 2026-06-08 got {s2.get('appointment_date')}")
if s2.get("appointment_time") not in [None, ""]:
    errors.append(f"T2 appointment_time should clear after date change, got {s2.get('appointment_time')}")
if (b2.get("pending") or {}).get("time"):
    errors.append("T2 booking.pending.time should clear after date change")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print(json.dumps({
  "scenario": "date_change",
  "old_date": s1.get("appointment_date"),
  "new_date": s2.get("appointment_date"),
  "stage": b2.get("stage"),
  "slots_found": s2.get("slots_found")
}, ensure_ascii=False, indent=2))
PY
  pass "Scenario C date change does not keep stale slot"
}

scenario_idempotency_and_persistence() {
  log "=== 6) Scenario D: idempotency + persisted state on changed-slot conversation ==="

  # Use a fresh changed-slot conversation so the duplicate final answer is stable.
  local conversation_id="advanced_idempotency_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_advanced_${conversation_id}"
  mkdir -p "$out_dir"

  clear_conversation "$conversation_id" "$out_dir"

  send_msg "$conversation_id" "001" "انا ساكن في العبور قولي اقرب فرع والمواعيد المتاحة يوم 2026-06-07" "$out_dir/turn_001.json"
  send_msg "$conversation_id" "002" "خليها الساعة 5" "$out_dir/turn_002.json"
  send_msg "$conversation_id" "003" "اه كمل" "$out_dir/turn_003.json"
  send_msg "$conversation_id" "004" "اسمي عبدالرحمن باسم" "$out_dir/turn_004.json"
  send_msg "$conversation_id" "005" "رقم العربيه ب ج د ٥٥٥" "$out_dir/turn_005.json"

  # Replay exact final idempotency key.
  curl -s \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -X POST "$API_URL/chat" \
    -d "{
      \"assistant_id\": \"$ASSISTANT_ID\",
      \"user_id\": \"$USER_ID\",
      \"conversation_id\": \"$conversation_id\",
      \"message\": \"رقم العربيه ب ج د ٥٥٥\",
      \"channel\": \"advanced_regression\",
      \"message_id\": \"${conversation_id}_005\",
      \"idempotency_key\": \"${conversation_id}_005\",
      \"debug\": true
    }" > "$out_dir/turn_005_duplicate.json"

  json_ok "$out_dir/turn_005_duplicate.json"

  curl -s \
    -H "x-api-key: $API_KEY" \
    "$API_URL/conversations/$ASSISTANT_ID/$conversation_id" > "$out_dir/conversation_state.json"

  json_ok "$out_dir/conversation_state.json"

  python3 - "$out_dir" <<'PY'
import json, sys
from pathlib import Path

out=Path(sys.argv[1])
orig=json.load(open(out/"turn_005.json", encoding="utf-8"))
dup=json.load(open(out/"turn_005_duplicate.json", encoding="utf-8"))
conv=json.load(open(out/"conversation_state.json", encoding="utf-8"))
vars=conv.get("variables") or {}
booking=vars.get("booking") or {}
profile=vars.get("customer_profile") or {}
errors=[]

if dup.get("duplicate") is not True:
    errors.append("duplicate replay did not return duplicate=true")
if dup.get("answer") != orig.get("answer"):
    errors.append("duplicate answer does not match original")
if booking.get("stage")!="confirmed":
    errors.append("persisted booking.stage not confirmed")
if vars.get("appointment_time")!="17:00":
    errors.append(f"persisted appointment_time expected 17:00 got {vars.get('appointment_time')}")
if profile.get("full_name")!="عبدالرحمن باسم":
    errors.append("persisted full_name wrong")
if profile.get("plate_number")!="ب ج د 555":
    errors.append("persisted plate wrong")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print(json.dumps({
  "scenario": "idempotency_persistence",
  "duplicate": dup.get("duplicate"),
  "stage": booking.get("stage"),
  "time": vars.get("appointment_time"),
  "plate": profile.get("plate_number")
}, ensure_ascii=False, indent=2))
PY
  pass "Scenario D idempotency and persisted state"
}

scenario_semantic_config() {
  log "=== 7) Scenario E: semantic extraction config and no-hardcoding sanity ==="

  local cid
  cid="$(container_id || true)"

  if [ -z "$cid" ]; then
    fail "semantic config: no Docker container"
    return 1
  fi

  docker exec -i "$cid" python3 - <<'PY'
import json

path="/app/configs/service_center_agentic_rag/domain_bundle.json"
bundle=json.load(open(path, encoding="utf-8"))
assistant=bundle.get("assistant") or {}
sem=assistant.get("semantic_variable_extraction") or {}
assert sem.get("enabled") is True, sem

fields=sem.get("fields") or []
ids={f.get("id") for f in fields if isinstance(f, dict)}
required={"customer_full_name","customer_phone","customer_plate_number"}
missing=required-ids
assert not missing, missing

for field in fields:
    if not isinstance(field, dict):
        continue
    if field.get("id") in required:
        for key in ["target_path","description","output_format","validation_description"]:
            assert field.get(key), (field.get("id"), key, field)
        assert field.get("required_when_stage_path") or field.get("required_when_paths"), field

answer_safety=assistant.get("answer_safety") or {}
assert answer_safety.get("record_id_label"), answer_safety
assert answer_safety.get("record_id_format"), answer_safety

fallback=assistant.get("fallback_messages") or {}
assert fallback.get("graph_error"), fallback
assert fallback.get("default_final"), fallback

print("semantic/no-hardcoding config ok")
PY

  pass "Scenario E semantic config sanity"
}

summary() {
  log "=== Advanced regression summary ==="
  echo "Passes: $pass_count"
  echo "Failures: $fail_count"

  if [ "$fail_count" -ne 0 ]; then
    echo ""
    echo "ADVANCED REGRESSION FAILED"
    echo "Inspect /root/agentic_advanced_* directories for debug JSON."
    exit 1
  fi

  echo ""
  echo "ALL ADVANCED REGRESSION CHECKS PASSED"
}

validate_deployment
validate_health
scenario_baseline_booking
scenario_hesitation_and_change_mind
scenario_date_change_after_slots
scenario_idempotency_and_persistence
scenario_semantic_config
summary
