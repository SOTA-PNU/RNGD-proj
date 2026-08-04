#!/usr/bin/env python3
"""chat_app.py 의 CATALOG 를 실제 자산·CLI 파서·serve_models.sh 와 교차 검증한다.

카탈로그를 고친 뒤 반드시 돌릴 것.  사용:  python3 validate_catalog.py
검사 항목:
  · 표시이름·포트 중복
  · 로컬 아티팩트 실재 + ctx/tp 가 artifact.json 실측치와 일치하는지
  · 프리빌트가 HF 캐시에 있는지, furiosa-ai/* 인지
  · FXB 번들인지 ↔ no_pp 설정이 맞는지 (FXB 에 -pp 주면 PanicException)
  · tool/reasoning 파서를 지금 설치된 furiosa-llm CLI 가 실제로 받는지
  · serve_models.sh 의 CAT 과 포트·파서·카드수·-pp 가 일치하는지
gradio 등 UI 의존성 없이 AST 로만 읽으므로 chat/.venv 없이도 돈다.
"""
import ast, json, os, re, subprocess, sys

# 인자를 안 주면 같은 폴더의 chat_app.py 를 본다.
CHAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_app.py")
SH = os.path.join(os.path.dirname(CHAT), "serve_models.sh")
ART = "/mnt/nvme2n1p1/models/artifacts"
HUB = "/mnt/nvme2n1p1/models/hf/hub"
FXB = os.path.expanduser("~/.cache/furiosa/llm/fxb")

# serve 런타임이 커널 없음으로 거부하는 (model_type, weight 양자화) 조합.
# 게이트는 hf_compat_next_gen.rs 의 화이트리스트이고, 연산 자체는 빌드 때 이미 컴파일돼
# 있으므로 masquerade_artifact.py 로 model_type 만 바꾸면 통과한다.
# 2026-08-04 실측: coder-tp8(qwen3_moe×fp8) → PanicException. 같은 qwen3_moe 라도 bf16 은 통과.
GATE_REJECTS = {("qwen3_moe", "fp8")}

# ── CATALOG 추출 (임포트 없이 AST 로) ────────────────────────────────────
tree = ast.parse(open(CHAT).read())
cat = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CATALOG":
        cat = {}
        for k, v in zip(node.value.keys, node.value.values):
            d = {}
            for kw in v.keywords:
                try:
                    d[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    d[kw.arg] = "<expr>"
            cat[ast.literal_eval(k)] = d
assert cat, "CATALOG 를 못 찾음"

errs, warns = [], []

# ── CLI 가 실제로 받는 파서 목록 ─────────────────────────────────────────
help_txt = subprocess.run([os.path.expanduser("~/furiosa/bin/furiosa-llm"), "serve", "--help"],
                          capture_output=True, text=True, timeout=300).stdout
def choices(flag):
    m = re.search(r"--%s \{([^}]*)\}" % flag, help_txt)
    return set(m.group(1).split(",")) if m else set()
TOOLS, REAS = choices("tool-call-parser"), choices("reasoning-parser")
print(f"CLI tool-call-parser  : {sorted(TOOLS)}")
print(f"CLI reasoning-parser  : {sorted(REAS)}\n")

names, ports = {}, {}
for k, m in cat.items():
    tag = f"[{k}]"
    # 이름·포트 유일성
    if m["name"] in names:
        errs.append(f"{tag} 이름 중복: {m['name']} (={names[m['name']]})")
    names[m["name"]] = k
    if m["port"] in ports:
        errs.append(f"{tag} 포트 중복: {m['port']} (={ports[m['port']]})")
    ports[m["port"]] = k

    src = m.get("src", "art")
    # ── 아티팩트 실재 + ctx 실측 대조 ───────────────────────────────────
    if src == "art":
        p = os.path.join(ART, m["sub"], "artifact.json")
        if not os.path.isfile(p):
            errs.append(f"{tag} artifact 없음: {p}")
        else:
            d = json.load(open(p))
            sizes = []
            def walk(o):
                if isinstance(o, dict):
                    for kk, vv in o.items():
                        if kk == "attention_size" and isinstance(vv, int):
                            sizes.append(vv)
                        walk(vv)
                elif isinstance(o, list):
                    for vv in o:
                        walk(vv)
            walk(d["model"]["pipeline_metadata_list"])
            real_ctx = max(sizes)
            if real_ctx != m["ctx"]:
                errs.append(f"{tag} ctx 불일치: 카탈로그 {m['ctx']} vs artifact {real_ctx}")
            tp = d["model"]["parallel_config"]["tensor_parallel_size"]
            if tp != m.get("pe", 8):
                errs.append(f"{tag} tp 불일치: 카탈로그 pe={m.get('pe', 8)} vs artifact tp={tp}")
            if m.get("no_pp"):
                errs.append(f"{tag} 로컬 v2 아티팩트인데 no_pp — pp 가능하므로 잘못됨")
            # serve 게이트: (model_type × 양자화) 조합에 런타임 커널이 없으면 부팅 시
            # PanicException: Unsupported model metadata 로 죽는다. 2026.3.0 에서도
            # qwen3_moe × FP8 은 여전히 거부된다(2026-08-04 실측). 위장으로 통과시킨다.
            mm = d["model"]["model_metadata"]
            mt = mm.get("model_type")
            wq = (mm.get("llm_config", {}).get("quantization_config") or {}).get("weight")
            if (mt, wq) in GATE_REJECTS:
                errs.append(
                    f"{tag} serve 게이트가 거부하는 조합 (model_type={mt} × weight={wq}) — "
                    f"masquerade 필요:\n"
                    f"        python3 ../../qwen3-next-proj/masquerade_artifact.py "
                    f"{os.path.join(ART, m['sub'])} --as qwen3 --in-place")
    else:
        repo = m["sub"]
        d = os.path.join(HUB, "models--" + repo.replace("/", "--"))
        if not os.path.isdir(d):
            errs.append(f"{tag} 프리빌트 저장소가 캐시에 없음: {repo}")
        if not repo.startswith("furiosa-ai/"):
            errs.append(f"{tag} furiosa-ai/* 가 아님(서빙 불가 가능성): {repo}")
        # FXB 여부 ↔ no_pp 일치
        is_fxb = os.path.isdir(os.path.join(FXB, "models--" + repo.replace("/", "--")))
        if m["kind"] == "tp8":
            if is_fxb and not m.get("no_pp"):
                errs.append(f"{tag} FXB 인데 no_pp 미설정 — pp 주면 PanicException")
            if not is_fxb and m.get("no_pp"):
                warns.append(f"{tag} v2 아티팩트인데 no_pp — pp 를 굳이 막고 있음")

    # ── 파서 유효성 ─────────────────────────────────────────────────────
    if m.get("tool") and m["tool"] not in TOOLS:
        errs.append(f"{tag} tool 파서 '{m['tool']}' 를 CLI 가 모름 → serve 즉시 실패")
    if m.get("reasoning") and m["reasoning"] not in REAS:
        errs.append(f"{tag} reasoning 파서 '{m['reasoning']}' 를 CLI 가 모름 → serve 즉시 실패")

    # ── pp/카드 예산 ────────────────────────────────────────────────────
    ppm = m.get("pp_min", 1)
    if m["kind"] == "tp8" and ppm > 4:
        errs.append(f"{tag} pp_min={ppm} > 카드 4장")
    if m.get("prompt_max") and m["prompt_max"] > m["ctx"]:
        errs.append(f"{tag} prompt_max({m['prompt_max']}) > ctx({m['ctx']})")

# ── serve_models.sh 와 포트·파서 대조 ────────────────────────────────────
sh = open(SH).read()
sh_entries = dict(re.findall(r"^\s*\[([A-Za-z0-9._-]+)\]=\"(.*?)\"\s*$", sh, re.M))
sh_entries.pop("hub-qwen2.5-0.5b_PE", None)
for k, m in cat.items():
    if k not in sh_entries:
        errs.append(f"[{k}] serve_models.sh 에 없음")
        continue
    port, cards, art, extra = sh_entries[k].split("|", 3)
    if int(port) != m["port"]:
        errs.append(f"[{k}] 포트 불일치: py {m['port']} vs sh {port}")
    if m.get("tool") and f"--tool-call-parser {m['tool']}" not in extra:
        errs.append(f"[{k}] tool 파서 불일치: py {m['tool']} vs sh '{extra}'")
    if m.get("reasoning") and f"--reasoning-parser {m['reasoning']}" not in extra:
        errs.append(f"[{k}] reasoning 파서 불일치: py {m['reasoning']} vs sh '{extra}'")
    exp_pp = m.get("pp_min", 1)
    if m["kind"] == "tp8" and exp_pp > 1 and f"-pp {exp_pp}" not in extra:
        errs.append(f"[{k}] sh 에 -pp {exp_pp} 없음: '{extra}'")
    exp_cards = 4 if m["kind"] == "tp32" else exp_pp
    if int(cards) != exp_cards:
        errs.append(f"[{k}] 카드수 불일치: 기대 {exp_cards} vs sh {cards}")
extra_keys = set(sh_entries) - set(cat)
if extra_keys:
    errs.append(f"serve_models.sh 에만 있는 키: {sorted(extra_keys)}")

print(f"모델 {len(cat)}종 검사 완료.")
for w in warns:
    print("  ⚠️ ", w)
for e in errs:
    print("  ❌ ", e)
print(("\n✅ 전부 통과" if not errs else f"\n❌ {len(errs)}건 실패"))
sys.exit(1 if errs else 0)
