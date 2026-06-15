# 버킷 preset 레지스트리(presets.py) 정리·검증

대상 파일: `~/furiosa/lib/python3.12/site-packages/furiosa_llm/artifact/presets.py`

`furiosa-llm build`에 버킷을 따로 안 주면(빈 `BucketConfig`), SDK가 모델의
`(model_type, hidden_size, intermediate_size)`를 보고 이 파일의 preset 중 맞는 걸 골라
씁니다. 그 매칭이 각 모델에 제대로 형성돼 있는지 실제 HF config와 코드로 검증한 결과를
정리합니다.

## preset가 고르는 방식

- `find_preset(model_type, hidden_size, intermediate_size)` (presets.py:363-390): 먼저
  `model_type`으로 후보를 거르고, 그중 layer당 파라미터 수(`approx_per_layer_params_b`,
  hidden·intermediate만 쓰는 함수)의 로그 거리가 가장 가까운 항목을 고릅니다.
- 빌드할 때 `--max-model-len`을 안 주면 그 값은 모델 config의 `max_position_embeddings`로
  잡힙니다(resolver.py:147-150의 `resolve_max_model_len`). 그래서 preset 안에서 그 길이를
  넘는 버킷은 `filter_preset_by_max_model_len`이 잘라냅니다(presets.py:404, `attention_size`
  기준). 잘라낸 뒤에도 남는 버킷이 `max_model_len`을 넘으면 `validate_resolved_buckets`가
  에러를 냅니다(validator.py:167-199, Rule 3·4).

즉 preset의 최대 컨텍스트가 모델보다 커도, 빌드 시 모델 길이에 맞춰 자동으로 잘립니다.

## 검증 1. 등록된 12개 키가 실제 모델과 맞는가

`PRESET_REFS`(presets.py:264-355)에 적힌 `(model_type, hidden_size, intermediate_size)`를
각 모델의 실제 HF `config.json`과 대조했습니다. 출처는 로컬 HF 캐시, 빌드된 아티팩트의
`config.json`, 그리고 huggingface.co의 `config.json`입니다(gated인 Llama는 unsloth 미러).

결과: 12개 항목 전부 실제 config와 정확히 일치합니다.

| 모델 | model_type / h / i | 실제 layer | native ctx | preset 최대 ctx | 컨텍스트 평가 |
|---|---|--:|--:|--:|---|
| Qwen2.5-0.5B-Instruct | qwen2 / 896 / 4864 | 24 | 32768 | 4096 | 일부러 낮춤(주석 없음) |
| Qwen2.5-Coder-1.5B | qwen2 / 1536 / 8960 | 28 | 32768 | 32768 | 일치 |
| Qwen2.5-Coder-7B (= Qwen2.5-7B) | qwen2 / 3584 / 18944 | 28 | 32768 | 32768 | 일치 |
| Qwen2.5-Coder-14B | qwen2 / 5120 / 13824 | 48 | 32768 | 32768 | 일치 |
| Qwen2.5-Coder-32B | qwen2 / 5120 / 27648 | 64 | 32768 | 32768 | 일치 |
| EXAONE-4.0-32B | exaone4 / 5120 / 27392 | 64 | 131072 | 131072 | 일치 |
| Llama-3.1-8B | llama / 4096 / 14336 | 32 | 131072 | 131072 | 일치 |
| Llama-3.3-70B | llama / 8192 / 28672 | 80 | 131072 | 131072 | 일치 |
| Qwen3-32B | qwen3 / 5120 / 25600 | 64 | 40960 | 40960 | 일치 |
| Qwen3-Embedding-8B (pooling) | qwen3 / 4096 / 12288 | 36 | 40960 | 8192(prefill만) | 일부러 낮춤(주석 있음) |
| Qwen3-30B-A3B / Coder-30B-A3B | qwen3_moe / 2048 / 6144 | 48 | 40960 / 262144 | 262144 | 아래 주의점 참고 |
| Qwen3-Coder-480B-A35B | qwen3_moe / 6144 / 8192 | 62 | 262144 | 32768 | 일부러 낮춤(주석 있음) |

대부분 preset 최대 컨텍스트가 모델 native와 똑같습니다. 일부러 낮춘 세 가지(임베딩 8192,
480B 32768, 0.5B 4096) 중 앞 둘은 코드 주석에 이유가 적혀 있고, 0.5B만 설명이 없습니다.

## 검증 2. 매칭이 각 모델을 제 preset로 보내는가

- 등록된 12개 모델은 자기 `(h, i)`가 정확히 들어가 있어 로그 거리 0이 되므로, 항상 자기
  preset로 갑니다. model_type별로 layer당 파라미터 값이 전부 달라(중복 없음) 충돌도
  없습니다. 실제로 `find_preset`을 등록값으로 호출해 12개 모두 자기 preset이 나오는 걸
  확인했습니다.
- 등록 안 된 비슷한 모델은 가까운 preset로 떨어집니다. 예를 들어 Qwen3-14B는 Qwen3-32B
  preset(40960), Qwen2.5-3B·14B·72B는 Qwen2.5-Coder preset(32768), Llama-3.2-1B·3B는
  Llama-3.1-8B preset(131072)로 가는데 모두 합리적입니다.

## 주의점 (검증하며 발견)

1. **생성용 작은 Qwen3(dense)는 임베딩 preset로 잘못 감 → 빌드가 에러로 막힘.**
   `find_preset`은 task를 안 봅니다(resolver.py:217-221에서 task를 안 넘김). qwen3 계열
   중 작은 항목은 임베딩용 `QWEN_3_8B_POOLING_PRESET`(prefill만 있고 decode 버킷 없음,
   presets.py:199-202) 하나뿐이라, 생성용 Qwen3-8B(h=4096·i=12288, 임베딩판과 치수가
   같음)·Qwen3-4B·1.7B·0.6B를 빈 버킷으로 빌드하면 이 pooling preset로 매칭됩니다.
   다만 조용히 깨진 아티팩트가 나오는 게 아니라, `validate_resolved_buckets`가
   "Generative models require at least one decode bucket"으로 빌드를 막습니다
   (validator.py:154-158). 이 모델들을 생성용으로 빌드하려면 `-pb`/`-db`로 버킷을 직접
   줘야 합니다.

2. **gpt2·gpt_oss는 preset이 없음.** `PRESET_REFS`에 항목이 없어 `find_preset`이 None을
   돌려주고, 빌드는 "No bucket configuration provided and no matching bucket preset found"
   에러를 냅니다(resolver.py:90-93). 버킷을 직접 줘야 빌드됩니다. (이 둘은 SDK에서도
   미검증 model_type입니다.)

3. **MoE를 dense 공식으로 매칭함.** `approx_per_layer_params_b`는 hidden·intermediate만
   쓰는 dense 근사라 `num_experts`를 무시합니다. 등록된 qwen3_moe 2개를 구분하는 데는
   문제가 없지만, MoE 매칭이 dense 기준이라는 점은 알아둘 필요가 있습니다.

4. **Qwen2.5-0.5B preset은 4096까지만.** 모델 native는 32768인데 preset은 4096이 최대라
   기본 빌드 시 4K로 제한됩니다(presets.py:70-74). 못 넘을 뿐 오작동은 아니지만, 왜
   낮췄는지 주석이 없습니다.

5. **죽은 상수 하나.** `QWEN_3_30B_A3B_PRESET`(presets.py:207-215, 최대 8192)는 정의는
   돼 있으나 `PRESET_REFS`에 등록돼 있지 않습니다. 주석(presets.py:336-340)에 따르면
   30B-A3B를 256K Coder preset으로 통일하면서 일부러 뺀 것입니다.

6. **30B-A3B 주석이 컨텍스트를 뭉뚱그림.** presets.py:337 주석은 "Coder·일반 모두 native
   262144"라고 적었지만, 기본 repo `Qwen/Qwen3-30B-A3B`의 config는 `max_position_embeddings`가
   40960입니다(262144는 Coder판과 2507판). 둘이 `(h, i)`가 같아 같은 262144 preset을
   공유하는데, 기능상 문제는 없습니다. 일반 40960 모델로 빌드하면 `max_model_len`이
   40960으로 잡혀 262144 버킷이 자동으로 잘리기 때문입니다(위 "preset가 고르는 방식"
   참고). 다만 주석의 사실 표현은 부정확합니다.

## 한 줄 요약

등록된 12개 preset은 키(model_type·hidden·intermediate)와 컨텍스트가 모두 실제 모델에
맞게 형성돼 있습니다. 다만 task를 안 보는 매칭 특성 때문에 (a) 생성용 작은 Qwen3와
(b) preset 없는 gpt2/gpt_oss는 빈 버킷 자동 빌드가 막히고, 이때는 버킷을 직접 줘야 합니다.
이 실패들은 모두 에러로 분명히 드러나며 조용히 잘못된 결과를 내지는 않습니다.
