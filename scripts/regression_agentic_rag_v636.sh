#!/usr/bin/env bash
set -Eeuo pipefail

# Agentic RAG regression suite for:
# graph.py v6.36
# domain_bundle.json v6.31
# booking_subagent.py v6.30
# main.py v6.33
# config.py v6.34
#
# Usage:
#   export API_URL="http://localhost:8010"
#   export API_KEY="YOUR_API_KEY"
#   bash regression_agentic_rag_v636.sh
#
# Optional:
#   ASSISTANT_ID="service_center_agentic_rag" bash regression_agentic_rag_v636.sh

API_URL="${API_URL:-http://localhost:8010}"
API_KEY="${API_KEY:-${APP_SECRET:-}}"
ASSISTANT_ID="${ASSISTANT_ID:-service_center_agentic_rag}"
USER_ID="${USER_ID:-201554354929@s.whatsapp.net}"
CONVERSATION_ID="${CONVERSATION_ID:-regression_v636_$(date +%s)}"
OUT_DIR="${OUT_DIR:-/root/agentic_regression_${CONVERSATION_ID}}"

if [ -z "$API_KEY" ]; then
  echo "FAIL: API_KEY or APP_SECRET must be exported before running this script."
  echo "Example: export API_KEY='your-api-key'"
  exit 1
fi

mkdir -p "$OUT_DIR"

pass_count=0
fail_count=0

log() {
  printf '\n%s\n' "$*"
}

pass() {
  pass_count=$((pass_count + 1))
  echo "PASS: $*"
}

fail() {
  fail_count=$((fail_count + 1))
  echo "FAIL: $*"
}

assert_json_file() {
  local file="$1"
  if python3 -m json.tool "$file" >/dev/null 2>&1; then
    pass "valid JSON: $file"
  else
    fail "invalid JSON: $file"
    echo "Raw content:"
    cat "$file" || true
  fi
}

assert_contains() {
  local file="$1"
  local needle="$2"
  local label="$3"

  if grep -q "$needle" "$file"; then
    pass "$label"
  else
    fail "$label"
  fi
}

docker_container_id() {
  docker ps --filter "name=rag-api" -q | head -n 1
}

run_container_validation() {
  log "=== 1) Deployed file/version validation ==="

  local cid
  cid="$(docker_container_id || true)"

  if [ -z "$cid" ]; then
    fail "rag-api Docker container found"
    return
  fi

  pass "rag-api Docker container found: $cid"

  docker exec -i "$cid" sh -lc '
set -eu

python3 -m py_compile app/graph.py
python3 -m py_compile app/subagents/booking_subagent.py
python3 -m py_compile app/main.py
python3 -m py_compile app/config.py
python3 -m json.tool /app/configs/service_center_agentic_rag/domain_bundle.json >/dev/null

grep -q "6.36-manifest-history-limit-no-hardcoding-graph" app/graph.py
grep -q "graph_extract_pending_required_details_from_patterns" app/graph.py
grep -q "semantic_extraction_node" app/graph.py
grep -q "pre_response_guardrail_node" app/graph.py
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
' > "$OUT_DIR/deployed_validation.txt" 2>&1 \
    && pass "deployed files match expected architecture markers" \
    || {
      fail "deployed files/version validation"
      cat "$OUT_DIR/deployed_validation.txt"
    }
}

health_check() {
  log "=== 2) Health and config endpoints ==="

  curl -s "$API_URL/health" > "$OUT_DIR/health.json"
  assert_json_file "$OUT_DIR/health.json"

  python3 - "$OUT_DIR/health.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
print("health ok:", d)
PY
  if [ $? -eq 0 ]; then pass "health endpoint ok"; else fail "health endpoint ok"; fi

  curl -s \
    -H "x-api-key: $API_KEY" \
    "$API_URL/config-source/$ASSISTANT_ID" > "$OUT_DIR/config_source.json"

  assert_json_file "$OUT_DIR/config_source.json"

  python3 - "$OUT_DIR/config_source.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("source") == "domain_bundle", d
assert d.get("assistant_found") is True, d
assert d.get("schema_found") is True, d
print("config source ok:", d)
PY
  if [ $? -eq 0 ]; then pass "config source uses domain_bundle"; else fail "config source uses domain_bundle"; fi
}

clear_conversation() {
  log "=== 3) Clear regression conversation ==="

  curl -s -X POST "$API_URL/conversations/$ASSISTANT_ID/$CONVERSATION_ID/clear" \
    -H "x-api-key: $API_KEY" > "$OUT_DIR/clear.json"

  assert_json_file "$OUT_DIR/clear.json"

  python3 - "$OUT_DIR/clear.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
assert d.get("cleared") is True, d
print("clear ok:", d)
PY
  if [ $? -eq 0 ]; then pass "conversation cleared"; else fail "conversation cleared"; fi

  local cid
  cid="$(docker_container_id || true)"
  if [ -n "$cid" ]; then
    docker exec -i "$cid" sh -lc "rm -f '/app/data/conversations/$ASSISTANT_ID/$CONVERSATION_ID.json' || true" >/dev/null 2>&1 || true
  fi
}

send_msg() {
  local mid="$1"
  local msg="$2"
  local out="$3"
  local key="${CONVERSATION_ID}_${mid}"

  curl -s \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -X POST "$API_URL/chat" \
    -d "{
      \"assistant_id\": \"$ASSISTANT_ID\",
      \"user_id\": \"$USER_ID\",
      \"conversation_id\": \"$CONVERSATION_ID\",
      \"message\": \"$msg\",
      \"channel\": \"regression\",
      \"message_id\": \"$key\",
      \"idempotency_key\": \"$key\",
      \"debug\": true
    }" > "$out"

  assert_json_file "$out"
}

run_5_turn_booking() {
  log "=== 4) Five-turn booking regression ==="

  send_msg "001" "انا ساكن في العبور قولي اقرب فرع واولي المواعيد المتاحة يوم 2026-06-07" "$OUT_DIR/turn_001.json"
  send_msg "002" "معاد الساعة 3 هيكون مناسب معايا" "$OUT_DIR/turn_002.json"
  send_msg "003" "اه" "$OUT_DIR/turn_003.json"
  send_msg "004" "اسمي عبدالرحمن باسم" "$OUT_DIR/turn_004.json"
  send_msg "005" "رقم العربيه ب ج د ٥٥٥" "$OUT_DIR/turn_005.json"

  python3 - "$OUT_DIR" <<'PY'
import json, re, sys
from pathlib import Path

out_dir = Path(sys.argv[1])

def load(name):
    return json.load(open(out_dir / name, encoding="utf-8"))

turns = [load(f"turn_00{i}.json") for i in range(1, 6)]

def state(turn):
    return (turn.get("debug") or {}).get("state_after") or {}

def booking(turn):
    return state(turn).get("booking") or {}

def profile(turn):
    return state(turn).get("customer_profile") or {}

def tool(turn):
    return (turn.get("debug") or {}).get("tool_result") or {}

errors = []

# Turn 1: nearest branch + slots.
s1 = state(turns[0])
b1 = booking(turns[0])
if s1.get("selected_branch") != "Nasr City":
    errors.append(f"T1 selected_branch expected Nasr City got {s1.get('selected_branch')}")
if s1.get("appointment_date") != "2026-06-07":
    errors.append(f"T1 appointment_date expected 2026-06-07 got {s1.get('appointment_date')}")
if s1.get("slots_found") is not True:
    errors.append("T1 slots_found expected True")
if len(s1.get("available_slots") or []) < 1:
    errors.append("T1 expected available slots")

# Turn 2: slot selected and pending saved.
s2 = state(turns[1])
b2 = booking(turns[1])
p2 = b2.get("pending") or {}
if s2.get("appointment_time") != "15:00":
    errors.append(f"T2 appointment_time expected 15:00 got {s2.get('appointment_time')}")
if p2.get("time") != "15:00":
    errors.append(f"T2 booking.pending.time expected 15:00 got {p2.get('time')}")
if b2.get("stage") != "awaiting_confirmation":
    errors.append(f"T2 booking.stage expected awaiting_confirmation got {b2.get('stage')}")

# Turn 3: confirmation captured.
s3 = state(turns[2])
b3 = booking(turns[2])
if s3.get("customer_confirmed_booking") is not True:
    errors.append("T3 customer_confirmed_booking expected True")
if b3.get("stage") != "awaiting_customer_details":
    errors.append(f"T3 booking.stage expected awaiting_customer_details got {b3.get('stage')}")
if s3.get("appointment_time") != "15:00":
    errors.append("T3 appointment_time should remain 15:00")

# Turn 4: name saved and not lost.
s4 = state(turns[3])
p4 = profile(turns[3])
bp4 = (booking(turns[3]).get("customer_profile") or {})
if p4.get("full_name") != "عبدالرحمن باسم":
    errors.append(f"T4 customer_profile.full_name expected عبدالرحمن باسم got {p4.get('full_name')}")
if bp4.get("full_name") != "عبدالرحمن باسم":
    errors.append(f"T4 booking.customer_profile.full_name expected عبدالرحمن باسم got {bp4.get('full_name')}")
if s4.get("appointment_time") != "15:00":
    errors.append("T4 appointment_time should remain 15:00")

# Turn 5: plate saved, stage confirmed, visit ID in answer, slot preserved.
t5 = turns[4]
s5 = state(t5)
b5 = booking(t5)
p5 = profile(t5)
bp5 = b5.get("customer_profile") or {}
pending5 = b5.get("pending") or {}
answer5 = t5.get("answer") or ""

if b5.get("stage") != "confirmed":
    errors.append(f"T5 booking.stage expected confirmed got {b5.get('stage')}")
if s5.get("appointment_time") != "15:00":
    errors.append(f"T5 appointment_time expected 15:00 got {s5.get('appointment_time')}")
if pending5.get("time") != "15:00":
    errors.append(f"T5 booking.pending.time expected 15:00 got {pending5.get('time')}")
if p5.get("full_name") != "عبدالرحمن باسم":
    errors.append(f"T5 customer_profile.full_name expected عبدالرحمن باسم got {p5.get('full_name')}")
if p5.get("plate_number") != "ب ج د 555":
    errors.append(f"T5 customer_profile.plate_number expected ب ج د 555 got {p5.get('plate_number')}")
if bp5.get("full_name") != "عبدالرحمن باسم":
    errors.append(f"T5 booking.customer_profile.full_name expected عبدالرحمن باسم got {bp5.get('full_name')}")
if bp5.get("plate_number") != "ب ج د 555":
    errors.append(f"T5 booking.customer_profile.plate_number expected ب ج د 555 got {bp5.get('plate_number')}")
if "VIS-" not in answer5:
    errors.append(f"T5 answer expected VIS- visit ID, got: {answer5}")

# Ensure bad old failure phrases are gone.
bad_phrases = [
    "هسألك عن مواعيد متاحة",
    "الحجز لسه مش مؤكد",
    "اختار معاد",
    "إيه المعاد اللي يناسبك",
]
for phrase in bad_phrases:
    if phrase in answer5:
        errors.append(f"T5 answer contains old failure phrase: {phrase}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

summary = {
    "selected_branch": s5.get("selected_branch"),
    "appointment_date": s5.get("appointment_date"),
    "appointment_time": s5.get("appointment_time"),
    "booking_stage": b5.get("stage"),
    "full_name": p5.get("full_name"),
    "plate_number": p5.get("plate_number"),
    "answer": answer5,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

  if [ $? -eq 0 ]; then
    pass "5-turn booking creates confirmed booking with visit ID"
  else
    fail "5-turn booking regression"
  fi
}

run_idempotency_test() {
  log "=== 5) Idempotency duplicate replay ==="

  local duplicate_out="$OUT_DIR/turn_005_duplicate.json"

  curl -s \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -X POST "$API_URL/chat" \
    -d "{
      \"assistant_id\": \"$ASSISTANT_ID\",
      \"user_id\": \"$USER_ID\",
      \"conversation_id\": \"$CONVERSATION_ID\",
      \"message\": \"رقم العربيه ب ج د ٥٥٥\",
      \"channel\": \"regression\",
      \"message_id\": \"${CONVERSATION_ID}_005\",
      \"idempotency_key\": \"${CONVERSATION_ID}_005\",
      \"debug\": true
    }" > "$duplicate_out"

  assert_json_file "$duplicate_out"

  python3 - "$OUT_DIR/turn_005.json" "$duplicate_out" <<'PY'
import json, sys

original = json.load(open(sys.argv[1], encoding="utf-8"))
duplicate = json.load(open(sys.argv[2], encoding="utf-8"))

assert duplicate.get("duplicate") is True, duplicate
assert duplicate.get("answer") == original.get("answer"), {
    "original": original.get("answer"),
    "duplicate": duplicate.get("answer"),
}
print("duplicate replay ok")
PY

  if [ $? -eq 0 ]; then pass "duplicate idempotency replay"; else fail "duplicate idempotency replay"; fi
}

run_conversation_state_test() {
  log "=== 6) Persisted conversation state endpoint ==="

  curl -s \
    -H "x-api-key: $API_KEY" \
    "$API_URL/conversations/$ASSISTANT_ID/$CONVERSATION_ID" > "$OUT_DIR/conversation_state.json"

  assert_json_file "$OUT_DIR/conversation_state.json"

  python3 - "$OUT_DIR/conversation_state.json" <<'PY'
import json, sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
variables = d.get("variables") or {}
booking = variables.get("booking") or {}
profile = variables.get("customer_profile") or {}

assert d.get("source") in {"postgres", "legacy_json"}, d.get("source")
assert booking.get("stage") == "confirmed", booking
assert profile.get("full_name") == "عبدالرحمن باسم", profile
assert profile.get("plate_number") == "ب ج د 555", profile
assert variables.get("appointment_time") == "15:00", variables
print("conversation state ok")
PY

  if [ $? -eq 0 ]; then pass "conversation state persisted"; else fail "conversation state persisted"; fi
}

run_semantic_config_test() {
  log "=== 7) Semantic extraction config sanity ==="

  local cid
  cid="$(docker_container_id || true)"

  if [ -z "$cid" ]; then
    fail "semantic config check: no Docker container"
    return
  fi

  docker exec -i "$cid" python3 - <<'PY'
import json

path = "/app/configs/service_center_agentic_rag/domain_bundle.json"
bundle = json.load(open(path, encoding="utf-8"))
assistant = bundle.get("assistant") or {}
sem = assistant.get("semantic_variable_extraction") or {}
assert sem.get("enabled") is True, sem

fields = sem.get("fields") or []
ids = {f.get("id") for f in fields if isinstance(f, dict)}
required = {"customer_full_name", "customer_phone", "customer_plate_number"}
missing = required - ids
assert not missing, missing

for field in fields:
    if not isinstance(field, dict):
        continue
    if field.get("id") in required:
        assert field.get("target_path"), field
        assert field.get("description"), field
        assert field.get("output_format"), field
        assert field.get("validation_description"), field

print("semantic config ok")
PY

  if [ $? -eq 0 ]; then pass "semantic extraction config sane"; else fail "semantic extraction config sane"; fi
}

run_summary() {
  log "=== Regression summary ==="
  echo "Output directory: $OUT_DIR"
  echo "Passes: $pass_count"
  echo "Failures: $fail_count"

  if [ "$fail_count" -ne 0 ]; then
    echo ""
    echo "FAILED. Inspect files in: $OUT_DIR"
    exit 1
  fi

  echo ""
  echo "ALL REGRESSION CHECKS PASSED"
}

run_container_validation
health_check
clear_conversation
run_5_turn_booking
run_idempotency_test
run_conversation_state_test
run_semantic_config_test
run_summary
