#!/usr/bin/env bash
set -Eeuo pipefail

# Advanced Agentic RAG smartness regression suite v3.
# Tests:
# - baseline deployment/health/config
# - v6.42 smartness runtime/config markers
# - dynamic slot booking with required vehicle details
# - hold/license behavior
# - field-help clarification
# - hesitation/advisor behavior
# - slot change / contradiction tracking
# - wrong-label vehicle detail extraction: year as class, class as model year
# - date change stale-slot clearing
# - semantic/bundle integrity
# - v6.45 code-expert cost/runtime controls and config-driven prompt compaction
#
# Usage:
#   export API_URL="http://localhost:8010"
#   export API_KEY="YOUR_API_KEY"
#   bash scripts/regression_agentic_rag_smartness_v3.sh

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
skip_count=0

pass(){ pass_count=$((pass_count+1)); echo "PASS: $*"; }
fail(){ fail_count=$((fail_count+1)); echo "FAIL: $*"; }
skip(){ skip_count=$((skip_count+1)); echo "SKIP: $*"; }
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

  curl -sS -X POST "$API_URL/conversations/$ASSISTANT_ID/$cid/clear" \
    -H "x-api-key: $API_KEY" > "$out_dir/clear.json"
  json_ok "$out_dir/clear.json" || return 1

  if python3 - "$out_dir/clear.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
assert d.get("cleared") is True, d
print("clear ok")
PY
  then
    pass "conversation cleared: $cid"
  else
    fail "conversation clear failed: $cid"
    return 1
  fi
}

send_msg(){
  local conversation_id="$1"
  local mid="$2"
  local msg="$3"
  local out="$4"
  local key="${conversation_id}_${mid}"
  local payload="${out}.payload.json"

  python3 - "$ASSISTANT_ID" "$USER_ID" "$conversation_id" "$msg" "$key" "$payload" <<'PY'
import json, sys
assistant_id, user_id, conversation_id, msg, key, payload = sys.argv[1:7]
body = {
    "assistant_id": assistant_id,
    "user_id": user_id,
    "conversation_id": conversation_id,
    "message": msg,
    "channel": "smartness_regression_v3",
    "message_id": key,
    "idempotency_key": key,
    "debug": True,
}
open(payload, "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY

  curl -sS \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -X POST "$API_URL/chat" \
    --data-binary "@$payload" > "$out"

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

py_validate(){
  local description="$1"
  shift
  if python3 "$@"; then
    pass "$description"
  else
    fail "$description"
    return 1
  fi
}

validate_deployment(){
  log "=== 1) Architecture, smartness, and no-hardcoding deployment validation ==="
  local cid
  cid="$(container_id || true)"
  if [ -z "$cid" ]; then fail "rag-api Docker container found"; return 1; fi
  pass "rag-api Docker container found: $cid"

  if docker exec -i "$cid" sh -lc '
set -eu

python3 -m py_compile app/graph.py
python3 -m py_compile app/subagents/booking_subagent.py
python3 -m py_compile app/main.py
python3 -m py_compile app/config.py
[ ! -f app/memory.py ] || python3 -m py_compile app/memory.py
python3 -m json.tool /app/configs/service_center_agentic_rag/domain_bundle.json >/dev/null

# Core graph/runtime markers.
grep -q "6.45-code-expert-cost-smartness-no-hardcoding-graph" app/graph.py
grep -q "6.36-manifest-history-limit-no-hardcoding-graph" app/graph.py
grep -q "6.39-previous-manifest-summary-no-hardcoding-graph" app/graph.py
grep -q "6.42-breathtaking-smartness-runtime-no-hardcoding-graph" app/graph.py
grep -q "6.44-semantic-detail-safety-no-hardcoding-graph" app/graph.py

# v6.42 runtime surfaces.
grep -q "best_guess_clarification" app/graph.py
grep -q "compute_reply_length_from_message" app/graph.py
grep -q "apply_length_mirroring" app/graph.py
grep -q "emotion_history" app/graph.py
grep -q "emotion_trajectory" app/graph.py
grep -q "compute_emotion_trajectory" app/graph.py
grep -q "apply_hesitation_detection" app/graph.py
grep -q "last_offered_options" app/graph.py
grep -q "apply_implicit_confirmation_guardrail" app/graph.py
grep -q "funnel_stage" app/graph.py
grep -q "variable_changes_this_turn" app/graph.py
grep -q "build_opener_context" app/graph.py
grep -q "proactive_surface_items" app/graph.py
grep -q "build_proactive_surface_items" app/graph.py
grep -q "apply_memory_to_variable_bridge" app/graph.py
grep -q "smart_inference_node" app/graph.py
grep -q "validate_and_heal_variables" app/graph.py
grep -q "run_batch_semantic_extraction" app/graph.py
grep -q "sanitize_semantic_extracted_value" app/graph.py
grep -q "build_failure_recovery_context" app/graph.py
grep -q "build_progressive_display_context" app/graph.py
grep -q "stuck_pattern" app/graph.py
grep -q "detect_stuck_pattern" app/graph.py

# v6.45 code-expert cost/runtime surfaces.
grep -q "graph_env_int" app/graph.py
grep -q "SUBAGENT_REASONING_MAX_TOKENS" app/graph.py
grep -q "compact_variables_for_response" app/graph.py
grep -q "configured_response_compaction" app/graph.py
grep -q "get_response_model" app/graph.py
grep -q "configured_response_model_routing" app/graph.py
grep -q "should_retry_full_manifest_for_stripped_updates" app/graph.py
grep -q "get_manifest_retry_policy" app/graph.py
grep -q "should_skip_subagent_reasoning_after_tool" app/graph.py
grep -q "build_response_guidance_block" app/graph.py
grep -q "quality_guard_policy" app/graph.py
! grep -q "max_tokens=700" app/graph.py

# Prior robust runtime markers.
grep -q "semantic_extraction_node" app/graph.py
grep -q "graph_extract_pending_required_details_from_patterns" app/graph.py
grep -q "WHAT IS ALREADY KNOWN" app/graph.py
grep -q "PREVIOUS TURN MANIFEST SUMMARY" app/graph.py
grep -q "MANIFEST_HISTORY_LIMIT" app/graph.py
! grep -q "messages\[-12:\]" app/graph.py

# Booking/main/config markers.
grep -q "6.30-skip-early-slot-guard-in-detail-stage-no-hardcoding" app/subagents/booking_subagent.py
grep -q "6.43-deterministic-vehicle-detail-correction-no-hardcoding" app/subagents/booking_subagent.py
grep -q "6.44-semantic-detail-safety-no-hardcoding" app/subagents/booking_subagent.py
grep -q "is_hold_or_delay_message" app/subagents/booking_subagent.py
grep -q "handle_field_help_if_needed" app/subagents/booking_subagent.py
grep -q "apply_customer_detail_cross_field_corrections" app/subagents/booking_subagent.py
grep -q "mirror_canonical_customer_profile_to_booking_profile" app/subagents/booking_subagent.py
grep -q "6.33-config-driven-main-error-handling-no-hardcoding" app/main.py
grep -q "6.34-runtime-controls-no-hardcoding" app/config.py
grep -q "6.45-code-expert-runtime-controls-no-hardcoding" app/config.py
grep -q "MODEL_RESPONSE_SIMPLE" app/config.py
grep -q "MAX_SUBAGENT_REASONING_TOKENS" app/config.py
grep -q "RESPONSE_MODEL_ROUTING_GLOBAL_ENABLED" app/config.py

# Bundle markers/config surfaces.
grep -q "6.31-semantic-variable-extraction-config-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.40-vehicle-details-and-varied-detail-prompts-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.41-field-help-hold-and-natural-vehicle-prompts-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.42-breathtaking-smartness-config-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.43-deterministic-vehicle-detail-correction-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.44-semantic-detail-safety-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "6.45-code-expert-cost-smartness-config-no-hardcoding" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "vehicle_detail_correction" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "value_must_appear_in_message" /app/configs/service_center_agentic_rag/domain_bundle.json

grep -q "\"smartness\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"smart_clarification\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"message_length_mirroring\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"emotion_arc_tracking\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"hesitation_detection\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"implicit_confirmation\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"last_offered_options\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"funnel_awareness\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"contradiction_acknowledgment\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"opener_context\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"proactive_surface\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"memory_to_variable_bridge\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"smart_inference\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"variable_schema_validation\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"tool_failure_recovery\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"progressive_display\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"stuck_pattern_detection\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"batch_extraction\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"manifest_context\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"manifest_retry_policy\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"response_model_routing\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"response_compaction\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"quality_guard_policy\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"subagent_reasoning_policy\"" /app/configs/service_center_agentic_rag/domain_bundle.json

# Vehicle required fields.
grep -q "\"car_brand\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"car_class\"" /app/configs/service_center_agentic_rag/domain_bundle.json
grep -q "\"car_model_year\"" /app/configs/service_center_agentic_rag/domain_bundle.json

# Guard against putting the newly added phrase packs in graph.py.
! grep -q "مش عارف" app/graph.py
! grep -q "استنى أجيب الرخصة" app/graph.py
! grep -q "يعني إيه الفئة" app/graph.py
'
  then
    pass "architecture markers, smartness surfaces, and no-hardcoding checks"
  else
    fail "architecture markers, smartness surfaces, and no-hardcoding checks"
    return 1
  fi
}

validate_health(){
  log "=== 2) API health/config validation ==="
  local out_dir="$OUT_ROOT/agentic_smartness_health"
  mkdir -p "$out_dir"

  curl -sS "$API_URL/health" > "$out_dir/health.json"
  json_ok "$out_dir/health.json" || return 1
  if python3 - "$out_dir/health.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("ok") is True, d
print("health ok")
PY
  then pass "health ok"; else fail "health ok"; return 1; fi

  curl -sS -H "x-api-key: $API_KEY" "$API_URL/config-source/$ASSISTANT_ID" > "$out_dir/config_source.json"
  json_ok "$out_dir/config_source.json" || return 1
  if python3 - "$out_dir/config_source.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("source") == "domain_bundle", d
assert d.get("assistant_found") is True, d
assert d.get("schema_found") is True, d
print("config source ok")
PY
  then pass "config-source ok"; else fail "config-source ok"; return 1; fi
}

scenario_full_booking_vehicle_help_hold(){
  log "=== 3) Scenario A: dynamic booking + hold/license + class help + vehicle details ==="
  local conversation_id="smart_full_vehicle_help_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_smartness_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir" || return 1

  send_msg "$conversation_id" "001" "انا ساكن في العبور قولي اقرب فرع واولي المواعيد المتاحة يوم $TEST_DATE" "$out_dir/turn_001.json" || return 1

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

  local chosen_text chosen_time
  chosen_text="$(slot_text "$out_dir/turn_001.json" 0)"
  chosen_time="$(slot_time "$out_dir/turn_001.json" 0)"
  echo "Scenario A selected slot: $chosen_text ($chosen_time)"

  send_msg "$conversation_id" "002" "معاد الساعة $chosen_text هيكون مناسب معايا" "$out_dir/turn_002.json" || return 1
  send_msg "$conversation_id" "003" "اه" "$out_dir/turn_003.json" || return 1
  send_msg "$conversation_id" "004" "استنى أجيب الرخصة" "$out_dir/turn_004.json" || return 1
  send_msg "$conversation_id" "005" "يعني إيه الفئة؟" "$out_dir/turn_005.json" || return 1
  send_msg "$conversation_id" "006" "اسمي عبدالرحمن باسم ورقم العربية ب ج د ٥٥٥ والعربية هيونداي النترا الفئة التانية موديل ٢٠٢٢" "$out_dir/turn_006.json" || return 1
  send_msg "$conversation_id" "007" "شكرا" "$out_dir/turn_007.json" || return 1

  if python3 - "$out_dir" "$chosen_time" <<'PY'
import json, sys, re
from pathlib import Path
out=Path(sys.argv[1])
expected=sys.argv[2]

def load(name): return json.load(open(out/name, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def manifest(t): return state(t).get("manifest") or {}
def booking(t): return state(t).get("booking") or {}
def profile(t): return state(t).get("customer_profile") or {}
def answer(t): return t.get("answer") or ""

turns=[load(f"turn_00{i}.json") for i in range(1,8)]
t1,t2,t3,t4,t5,t6,t7=turns
errors=[]

s1=state(t1)
if s1.get("selected_branch")!="Nasr City": errors.append(f"T1 branch expected Nasr City got {s1.get('selected_branch')}")
if s1.get("appointment_date")!="2026-06-07": errors.append(f"T1 appointment_date wrong: {s1.get('appointment_date')}")
if len(s1.get("available_slots") or []) < 1: errors.append("T1 expected available slots")
if not s1.get("last_offered_options"): errors.append("T1 expected last_offered_options after showing slots")

s2=state(t2); b2=booking(t2); p2=b2.get("pending") or {}
if s2.get("appointment_time")!=expected: errors.append(f"T2 time expected {expected} got {s2.get('appointment_time')}")
if p2.get("time")!=expected: errors.append(f"T2 pending.time expected {expected} got {p2.get('time')}")
if b2.get("stage")!="awaiting_confirmation": errors.append(f"T2 stage expected awaiting_confirmation got {b2.get('stage')}")

s3=state(t3); b3=booking(t3)
if s3.get("customer_confirmed_booking") is not True: errors.append("T3 customer_confirmed_booking expected true")
if b3.get("stage")!="awaiting_customer_details": errors.append(f"T3 stage expected awaiting_customer_details got {b3.get('stage')}")

s4=state(t4); p4=profile(t4); a4=answer(t4)
if booking(t4).get("stage")!="awaiting_customer_details": errors.append("T4 hold should keep awaiting_customer_details")
if p4.get("car_brand") or p4.get("car_class") or p4.get("car_model_year"): errors.append(f"T4 hold should not extract vehicle fields: {p4}")
if "VIS-" in a4: errors.append("T4 hold should not create booking")
if not any(x in a4 for x in ["خد وقتك", "مستنيك", "الرخصة", "جاهز", "براحتك"]): errors.append(f"T4 hold answer did not feel like wait/license handling: {a4}")

s5=state(t5); p5=profile(t5); a5=answer(t5)
if booking(t5).get("stage")!="awaiting_customer_details": errors.append("T5 field-help should keep awaiting_customer_details")
if p5.get("car_class"): errors.append(f"T5 question about class should not store car_class: {p5.get('car_class')}")
if not any(x in a5 for x in ["فئة", "نسخة", "trim", "هاي لاين", "C-Class", "الرخصة"]): errors.append(f"T5 class-help answer missing explanation: {a5}")

s6=state(t6); b6=booking(t6); p6=profile(t6); bp6=b6.get("customer_profile") or {}
a6=answer(t6)
expected_fields = {
    "full_name": "عبدالرحمن باسم",
    "plate_number": "ب ج د 555",
    "car_brand": "هيونداي",
    "car_class": "الفئة التانية",
    "car_model_year": "2022",
}
for key, val in expected_fields.items():
    got = p6.get(key)
    if got != val:
        errors.append(f"T6 customer_profile.{key} expected {val!r} got {got!r}")
    bgot = bp6.get(key)
    if bgot != val and key in ["full_name","plate_number","car_brand","car_class","car_model_year"]:
        errors.append(f"T6 booking.customer_profile.{key} expected {val!r} got {bgot!r}")
if b6.get("stage")!="confirmed": errors.append(f"T6 booking.stage expected confirmed got {b6.get('stage')}")
if s6.get("appointment_time")!=expected: errors.append(f"T6 appointment_time expected {expected} got {s6.get('appointment_time')}")
if "VIS-" not in a6: errors.append(f"T6 answer expected VIS id, got: {a6}")

a7=answer(t7)
if len(a7) > 220: errors.append(f"T7 thanks/closing answer too long ({len(a7)} chars): {a7}")
if "VIS-" in a7: errors.append("T7 should not repeat visit ID after thanks")

m1=manifest(t1)
if m1.get("funnel_stage") in [None, ""]: errors.append("Manifest should expose funnel_stage")
if "response_brief" in m1 and not isinstance(m1.get("response_brief"), dict): errors.append("Manifest response_brief should be dict")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print(json.dumps({
    "scenario": "full_booking_vehicle_help_hold",
    "stage": b6.get("stage"),
    "time": s6.get("appointment_time"),
    "profile": p6,
    "closing_answer_length": len(a7),
    "answer": a6
}, ensure_ascii=False, indent=2))
PY
  then
    pass "Scenario A dynamic booking with hold/help/vehicle details"
  else
    fail "Scenario A dynamic booking with hold/help/vehicle details"
    return 1
  fi
}

scenario_hesitation_change_mind_vehicle_swaps(){
  log "=== 4) Scenario B: hesitation + change mind + wrong-label vehicle detail correction ==="
  local conversation_id="smart_hesitation_change_vehicle_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_smartness_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir" || return 1

  send_msg "$conversation_id" "001" "انا ساكن في العبور بس مش متأكد، ممكن تقولي اقرب فرع والمواعيد المتاحة يوم $TEST_DATE؟" "$out_dir/turn_001.json" || return 1

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

  send_msg "$conversation_id" "002" "مش عارف اختار انهي معاد، ساعدني" "$out_dir/turn_002.json" || return 1
  send_msg "$conversation_id" "003" "خليها الساعة $first_text" "$out_dir/turn_003.json" || return 1
  send_msg "$conversation_id" "004" "لا استنى، خليها الساعة $second_text بدل المعاد الأول" "$out_dir/turn_004.json" || return 1
  send_msg "$conversation_id" "005" "اه كمل" "$out_dir/turn_005.json" || return 1
  send_msg "$conversation_id" "006" "الفئة ٢٠٢٢" "$out_dir/turn_006.json" || return 1
  send_msg "$conversation_id" "007" "الموديل هاي لاين" "$out_dir/turn_007.json" || return 1
  send_msg "$conversation_id" "008" "اسمي عبدالرحمن باسم ورقم العربيه ب ج د ٥٥٥ وماركة العربية هيونداي" "$out_dir/turn_008.json" || return 1

  if python3 - "$out_dir" "$first_time" "$second_time" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
first_expected=sys.argv[2]
second_expected=sys.argv[3]

def load(name): return json.load(open(out/name, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def manifest(t): return state(t).get("manifest") or {}
def booking(t): return state(t).get("booking") or {}
def profile(t): return state(t).get("customer_profile") or {}
def answer(t): return t.get("answer") or ""

turns=[load(f"turn_00{i}.json") for i in range(1,9)]
errors=[]

s2=state(turns[1]); m2=manifest(turns[1]); a2=answer(turns[1])
if s2.get("selected_branch")!="Nasr City": errors.append("T2 branch lost during hesitation")
if s2.get("appointment_date")!="2026-06-07": errors.append("T2 date lost during hesitation")
if len(s2.get("available_slots") or []) < 1: errors.append("T2 slots lost during hesitation")
if m2.get("customer_emotion") != "undecided" and "advisor" not in str((m2.get("response_brief") or {}).get("tone","")).lower():
    errors.append(f"T2 hesitation should set undecided/advisor, got emotion={m2.get('customer_emotion')} brief={m2.get('response_brief')}")
if not s2.get("emotion_history"): errors.append("T2 expected emotion_history to be present")
if not s2.get("emotion_trajectory"): errors.append("T2 expected emotion_trajectory to be present")

s3=state(turns[2]); p3=(booking(turns[2]).get("pending") or {})
if s3.get("appointment_time")!=first_expected: errors.append(f"T3 expected {first_expected} got {s3.get('appointment_time')}")
if p3.get("time")!=first_expected: errors.append(f"T3 pending expected {first_expected} got {p3.get('time')}")

s4=state(turns[3]); p4=(booking(turns[3]).get("pending") or {})
if s4.get("appointment_time")!=second_expected: errors.append(f"T4 expected changed time {second_expected} got {s4.get('appointment_time')}")
if p4.get("time")!=second_expected: errors.append(f"T4 pending expected {second_expected} got {p4.get('time')}")
if not s4.get("variable_changes_this_turn"): errors.append("T4 expected variable_changes_this_turn after slot change")

s5=state(turns[4]); b5=booking(turns[4])
if s5.get("customer_confirmed_booking") is not True: errors.append("T5 confirmation not true")
if s5.get("appointment_time")!=second_expected: errors.append("T5 changed slot not preserved")
if b5.get("stage")!="awaiting_customer_details": errors.append(f"T5 stage expected awaiting_customer_details got {b5.get('stage')}")

s6=state(turns[5]); p6=profile(turns[5])
if p6.get("car_model_year") != "2022": errors.append(f"T6 year-as-class should store car_model_year=2022, got {p6.get('car_model_year')}")
if p6.get("car_class") in ["2022", "٢٠٢٢"]: errors.append(f"T6 should not store year as car_class, got {p6.get('car_class')}")

s7=state(turns[6]); p7=profile(turns[6])
if p7.get("car_model_year") != "2022": errors.append(f"T7 should preserve car_model_year=2022, got {p7.get('car_model_year')}")
if "هاي" not in str(p7.get("car_class") or "") and "High" not in str(p7.get("car_class") or ""):
    errors.append(f"T7 class-as-model should store car_class, got {p7.get('car_class')}")

s8=state(turns[7]); b8=booking(turns[7]); p8=profile(turns[7]); bp8=b8.get("customer_profile") or {}
a8=answer(turns[7])
expected_fields = {
    "full_name": "عبدالرحمن باسم",
    "plate_number": "ب ج د 555",
    "car_brand": "هيونداي",
    "car_class": p7.get("car_class"),
    "car_model_year": "2022",
}
for key, val in expected_fields.items():
    got=p8.get(key)
    if got != val:
        errors.append(f"T8 customer_profile.{key} expected {val!r} got {got!r}")
    bgot=bp8.get(key)
    if bgot != val:
        errors.append(f"T8 booking.customer_profile.{key} expected {val!r} got {bgot!r}")
if b8.get("stage")!="confirmed": errors.append(f"T8 expected confirmed got {b8.get('stage')}")
if s8.get("appointment_time")!=second_expected: errors.append(f"T8 expected final time {second_expected} got {s8.get('appointment_time')}")
if "VIS-" not in a8: errors.append(f"T8 no visit ID in answer: {a8}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print(json.dumps({
    "scenario":"hesitation_change_mind_vehicle_swaps",
    "stage": b8.get("stage"),
    "first_time": first_expected,
    "final_time": s8.get("appointment_time"),
    "profile": p8,
    "answer": a8
}, ensure_ascii=False, indent=2))
PY
  then
    pass "Scenario B hesitation/change-mind/wrong-label vehicle details"
  else
    fail "Scenario B hesitation/change-mind/wrong-label vehicle details"
    return 1
  fi
}

scenario_date_change(){
  log "=== 5) Scenario C: date change clears stale slot ==="
  local conversation_id="smart_date_change_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_smartness_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir" || return 1

  send_msg "$conversation_id" "001" "انا ساكن في العبور قولي المواعيد المتاحة يوم 2026-06-07" "$out_dir/turn_001.json" || return 1
  send_msg "$conversation_id" "002" "لا مش اليوم ده، شوف يوم 2026-06-08" "$out_dir/turn_002.json" || return 1

  if python3 - "$out_dir" <<'PY'
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
if s2.get("variable_changes_this_turn") in [None, [], {}]: errors.append("T2 expected variable_changes_this_turn after date change")
if errors:
    print("\n".join(errors)); raise SystemExit(1)
print(json.dumps({"scenario":"date_change","old_date":s1.get("appointment_date"),"new_date":s2.get("appointment_date"),"stage":b2.get("stage"),"slots_found":s2.get("slots_found")}, ensure_ascii=False, indent=2))
PY
  then
    pass "Scenario C date change clears stale slot"
  else
    fail "Scenario C date change clears stale slot"
    return 1
  fi
}

scenario_smart_clarification_length(){
  log "=== 6) Scenario D: smart clarification + length mirroring smoke test ==="
  local conversation_id="smart_clarification_length_$(date +%s)"
  local out_dir="$OUT_ROOT/agentic_smartness_${conversation_id}"
  mkdir -p "$out_dir"
  clear_conversation "$conversation_id" "$out_dir" || return 1

  send_msg "$conversation_id" "001" "الخميس المعادي" "$out_dir/turn_001.json" || return 1
  send_msg "$conversation_id" "002" "تمام" "$out_dir/turn_002.json" || return 1

  if python3 - "$out_dir" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
def load(n): return json.load(open(out/n, encoding="utf-8"))
def state(t): return (t.get("debug") or {}).get("state_after") or {}
def manifest(t): return state(t).get("manifest") or {}
t1=load("turn_001.json"); t2=load("turn_002.json")
m1=manifest(t1); m2=manifest(t2)
a1=t1.get("answer") or ""; a2=t2.get("answer") or ""
errors=[]
if "best_guess_clarification" not in m1:
    errors.append("T1 manifest missing best_guess_clarification key")
bg=m1.get("best_guess_clarification") or {}
if isinstance(bg, dict):
    # For ambiguous message, we prefer a hypothesis if the model can form one.
    if m1.get("confidence", 1.0) < 0.8 and not bg.get("hypothesis"):
        errors.append(f"T1 low confidence but no hypothesis: {bg}")
if (m2.get("response_brief") or {}).get("reply_length") not in ["very_short", "short", ""]:
    errors.append(f"T2 short message should mirror to short/very_short, got {(m2.get('response_brief') or {}).get('reply_length')}")
if len(a2) > 260:
    errors.append(f"T2 short message answer too long ({len(a2)} chars): {a2}")
if errors:
    print("\n".join(errors))
    print(json.dumps({"t1_answer":a1, "t1_manifest":m1, "t2_answer":a2, "t2_manifest":m2}, ensure_ascii=False, indent=2))
    raise SystemExit(1)
print(json.dumps({
    "scenario": "smart_clarification_length",
    "t1_confidence": m1.get("confidence"),
    "best_guess": m1.get("best_guess_clarification"),
    "t2_reply_length": (m2.get("response_brief") or {}).get("reply_length"),
    "t2_answer_length": len(a2)
}, ensure_ascii=False, indent=2))
PY
  then
    pass "Scenario D smart clarification and length mirroring"
  else
    fail "Scenario D smart clarification and length mirroring"
    return 1
  fi
}

scenario_config_integrity(){
  log "=== 7) Scenario E: semantic, vehicle, and smartness config integrity ==="
  local cid
  cid="$(container_id || true)"
  if [ -z "$cid" ]; then fail "config integrity: no Docker container"; return 1; fi

  if docker exec -i "$cid" python3 - <<'PY'
import json
bundle=json.load(open("/app/configs/service_center_agentic_rag/domain_bundle.json", encoding="utf-8"))
assistant=bundle.get("assistant") or {}

sem=assistant.get("semantic_variable_extraction") or {}
assert sem.get("enabled") is True, sem
fields=sem.get("fields") or []
ids={f.get("id") for f in fields if isinstance(f, dict)}
required={
    "customer_full_name",
    "customer_phone",
    "customer_plate_number",
    "customer_car_brand",
    "customer_car_class",
    "customer_car_model_year",
}
missing=required-ids
assert not missing, missing
for field in fields:
    if isinstance(field, dict) and field.get("id") in required:
        for key in ["target_path","description","output_format","validation_description"]:
            assert field.get(key), (field.get("id"), key)

batch=(sem.get("batch_extraction") or {})
assert batch.get("enabled") is True, batch

booking=((assistant.get("subagents") or {}).get("booking") or {})

# Scenario E root rule #3 optional vehicle contract:
# create-blocking fields must match canonical create_booking.required_inputs.
# Optional vehicle details may be extracted/schema-backed/help-backed, but must
# not block create unless intentionally added to canonical tool required_inputs.
def _as_list(value):
    return [str(x) for x in value] if isinstance(value, list) else []

def _normalize_required_path(value):
    value = str(value or "").strip()
    aliases = {
        "variables.booking.pending.branch": "branch",
        "variables.booking.pending.date": "date",
        "variables.booking.pending.date_text": "date_text",
        "variables.booking.pending.time": "time",
        "variables.booking.pending.section": "section",
        "variables.customer_profile.full_name": "full_name",
        "variables.customer_profile.phone": "phone",
        "variables.customer_profile.plate_number": "plate_number",
        "variables.customer_confirmed_booking": "customer_confirmed_booking",
        "booking.pending.branch": "branch",
        "booking.pending.date": "date",
        "booking.pending.date_text": "date_text",
        "booking.pending.time": "time",
        "booking.pending.section": "section",
        "customer_profile.full_name": "full_name",
        "customer_profile.phone": "phone",
        "customer_profile.plate_number": "plate_number",
    }
    if value in aliases:
        return aliases[value]
    if "." in value:
        return value.split(".")[-1]
    return value

canonical_required = None
for entry in assistant.get("tool_catalog") or []:
    if isinstance(entry, dict) and entry.get("operation") == "create_booking":
        canonical_required = set(_as_list(entry.get("required_inputs")))
        break

assert canonical_required, "missing canonical create_booking.required_inputs"

required_before_create_raw = booking.get("required_before_create") or []
required_before_create = {
    _normalize_required_path(x)
    for x in _as_list(required_before_create_raw)
}

optional_vehicle = {"car_brand", "car_class", "car_model_year"}

assert not (canonical_required & optional_vehicle), {
    "optional_vehicle_in_canonical_required": sorted(canonical_required & optional_vehicle),
    "canonical_required": sorted(canonical_required),
}

assert not (required_before_create & optional_vehicle), {
    "optional_vehicle_in_required_before_create": sorted(required_before_create & optional_vehicle),
    "required_before_create_raw": required_before_create_raw,
}

assert required_before_create == canonical_required, {
    "required_before_create": sorted(required_before_create),
    "canonical_required": sorted(canonical_required),
}

field_help=booking.get("field_help") or {}
for p in [
    "customer_profile.car_brand",
    "customer_profile.car_class",
    "customer_profile.car_model_year",
]:
    assert p in field_help, p
    assert field_help[p].get("answer_variants"), p
    assert field_help[p].get("follow_up_variants"), p

hold=booking.get("hold_or_delay_response_policy") or {}
assert hold.get("enabled") is True, hold
assert hold.get("response_variants"), hold

smart=assistant.get("smartness") or {}
for key in [
    "smart_clarification",
    "message_length_mirroring",
    "emotion_arc_tracking",
    "hesitation_detection",
    "implicit_confirmation",
    "last_offered_options",
    "funnel_awareness",
    "contradiction_acknowledgment",
    "opener_context",
    "proactive_surface",
    "memory_to_variable_bridge",
    "smart_inference",
    "variable_schema_validation",
    "tool_failure_recovery",
    "progressive_display",
    "stuck_pattern_detection",
]:
    assert key in smart, key
    assert isinstance(smart[key], dict), key
    assert smart[key].get("enabled") is True or key in ["memory_to_variable_bridge"], (key, smart[key])

schema=(assistant.get("schema") or {}).get("variables") or {}
for p in [
    "customer_profile.full_name",
    "customer_profile.phone",
    "customer_profile.plate_number",
    "customer_profile.car_brand",
    "customer_profile.car_class",
    "customer_profile.car_model_year",
    "appointment_date",
]:
    assert p in schema, p

manifest_ctx=assistant.get("manifest_context") or {}
assert manifest_ctx.get("previous_manifest_summary_enabled") is True, manifest_ctx
assert int(manifest_ctx.get("previous_manifest_summary_max_chars") or 0) <= 600, manifest_ctx
assert manifest_ctx.get("previous_manifest_summary_fields"), manifest_ctx

retry=assistant.get("manifest_retry_policy") or {}
assert float(retry.get("source_of_truth_strip_ratio_threshold") or 0) >= 0.6, retry
assert int(retry.get("min_stripped_updates_for_retry") or 0) >= 3, retry

routing=assistant.get("response_model_routing") or {}
assert routing.get("enabled") is True, routing
assert routing.get("simple_model"), routing
assert routing.get("default_model"), routing
assert (routing.get("never_use_simple_model_when") or {}).get("tool_result") is True, routing

compaction=assistant.get("response_compaction") or {}
assert compaction.get("enabled") is True, compaction
assert compaction.get("exclude_variable_paths_when_tool_result_has"), compaction
assert int(compaction.get("max_variable_items") or 0) > 0, compaction

quality=assistant.get("quality_guard_policy") or {}
assert quality.get("enabled", True) is not False, quality
assert int(quality.get("long_answer_chars") or 0) >= 200, quality

subagent_reasoning=assistant.get("subagent_reasoning_policy") or {}
assert subagent_reasoning.get("enabled", True) is not False, subagent_reasoning
assert subagent_reasoning.get("skip_on_clean_executor_result") is True, subagent_reasoning
assert float(subagent_reasoning.get("min_manifest_confidence") or 0) >= 0.7, subagent_reasoning
assert subagent_reasoning.get("clean_actions"), subagent_reasoning

assert "response_model_routing" in smart and "response_compaction" in smart, smart

answer_safety=assistant.get("answer_safety") or {}
assert answer_safety.get("record_id_label"), answer_safety
assert answer_safety.get("record_id_format"), answer_safety
fallback=assistant.get("fallback_messages") or {}
assert fallback.get("graph_error"), fallback
assert fallback.get("default_final"), fallback

print("semantic/vehicle/smartness config ok")
PY
  then
    pass "Scenario E semantic/vehicle/smartness config integrity"
  else
    fail "Scenario E semantic/vehicle/smartness config integrity"
    return 1
  fi
}

summary(){
  log "=== Smartness regression v3 summary ==="
  echo "Passes: $pass_count"
  echo "Failures: $fail_count"
  echo "Skipped: $skip_count"
  if [ "$fail_count" -ne 0 ]; then
    echo ""
    echo "SMARTNESS REGRESSION V3 FAILED"
    echo "Inspect $OUT_ROOT/agentic_smartness_* directories for debug JSON."
    exit 1
  fi
  echo ""
  echo "ALL SMARTNESS REGRESSION V3 CHECKS PASSED"
}

validate_deployment
validate_health
scenario_full_booking_vehicle_help_hold
scenario_hesitation_change_mind_vehicle_swaps
scenario_date_change
scenario_smart_clarification_length
scenario_config_integrity
summary
