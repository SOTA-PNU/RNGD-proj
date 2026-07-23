# 논문에 쓰인 실험 결과 원본 모음

이 폴더는 ACCV 논문(`../main.md` · `../main_en.md`)의 **모든 표·그림·본문 수치**가 나온 실험 결과 원본을 한곳에 `cp`로 모은 것입니다(원본은 각 실험 폴더에 그대로 있습니다). 각 파일이 논문의 어느 요소에 대응하는지는 아래 표를 보세요.

**헤드라인 프로토콜**은 표준 train-갤러리 kNN(gallery=ImageNet train 1.28M, query=val 50k, 무압축 80.87≈공인 82)이고, val leave-one-out(val-LOO)은 일관성 확인용입니다. 검색 mAP는 자기검색이라 val, dense는 ADE20k입니다.

## 파일 → 논문 요소 매핑

| 파일 | 프로토콜 | 논문에서 쓰이는 곳 | 내용 |
|---|---|---|---|
| `canonical_faithful_base.txt` | train, ViT-B | **tab:main · tab:pitome · tab:generality**(ToMe/PiToMe/Ours 열) · fig_result(좌) · fig_pitome(좌) | 무압축 80.87; r별 tome/pitome/ours |
| `results_pitome_reg_train_faithful.txt` | train, ViT-B | **tab:generality**(PiToMe+reg 열, reg 이득) · §일반성 | PiToMe 병합 위 레지스터 보호(reg@PiTo +0.48→+5.07) |
| `ablation_train_faithful_base.txt` | train, ViT-B | **tab:ablation** · fig_ablation | ToMe/Ours/random/energy/high-norm |
| `reg_count_train_faithful_base.txt` | train, ViT-B | **tab:aux**(reg-count CI) · fig_result(우, reg-count 스윕) | k=0→4, 부트스트랩 95% CI |
| `canonical_faithful_small.txt` | train, ViT-S | §consistency(인코더 크기 확장, 무압축 77.41) | ViT-S tome/pitome/ours |
| `faithful_pitome_50k.log` | val-LOO, ViT-B | **tab:consistency**(val-LOO 3자) · 초록/한계 val 수치 | ToMe/PiToMe/Ours, 무압축 76.33 |
| `faithful_tome_50k.log` | val-LOO, ViT-B | §consistency(ToMe/Ours 교차확인) | 정식 ToMe vs Ours |
| `retrieval_map_faithful_50k.log` | val, ViT-B | **tab:aux**(검색 mAP 행) · §보조검증 | Ours−ToMe mAP +2.56→+16.40 |
| `reg_count_sweep_faithful_50k.log` | val-LOO, ViT-B | §consistency(val reg-count 재현) | k=0→4 val-LOO + CI |
| `ablation_faithful_50k.log` | val, ViT-B | §consistency(val ablation 재현) | 정식 ablation 5전략 |
| `dense_miou_FAITHFUL_vit_base_patch14_reg4_dinov2.json` | ADE20k, reg | **tab:dense**(레지스터 모델) | mIoU, 무압축 29.4 |
| `dense_miou_FAITHFUL_vit_base_patch14_dinov2.json` | ADE20k, no-reg | tab:dense(음성 대조) | 레지스터 없는 모델 dense |
| `dinov2_noreg_control.txt` | val, no-reg | **tab:control**(음성 대조군) | 레지스터 없는 DINOv2, Δ≈0 |
| `results_tput.txt` | 합성배치 | §효율 · fig_pitome(우, 처리량) | tome/pitome/ours im/s(~570대) |
| `flops.json` | 계산론적 | **tab:pitome**(FLOP 절감·GFLOPs 열) · §효율 | 무압축 23.5 GFLOPs, r별 절감 |
| `results_weighted_knn.txt` | train, 가중투표 | §consistency(투표방식 불변, 80.87→81.42) | 공식 온도가중 재채점 |
| `eval_v2_seeds3_50k.json` | val, 3-seed(통제) | §한계(다중 인코더 S/B/L·선형프로브·동적재선택) | 보조 3-seed 연구 |
| `deit_compare_report.txt` | DeiT, 공식 repo | §방법(이식 검증, 정직성) | 우리 포팅 vs 공식 PiToMe repo 대조 |
| `extra_dinov3_base_train_faithful.txt` | train, DINOv3-B | **tab:extra**(일반성) · §다른 인코더 계열 | Ours vs no-reg, 무압축 81.63 |
| `extra_dinov3_splus_train_faithful.txt` | train, DINOv3-S+ | **tab:extra** · §다른 인코더 계열 | Ours vs no-reg, 무압축 77.94 |
| `extra_vit5_base_train_faithful.txt` | train, ViT-5-B | **tab:extra** · §다른 인코더 계열 | Ours vs no-reg, 무압축 82.40 |
| `extra_dinov3_base_train_regsweep.txt` | train, DINOv3-B | §다른 인코더 계열(reg-count 스윕) | k=0→4, 1.77→54.75 단조증가 |
| `extra_vit5_base_train_regsweep.txt` | train, ViT-5-B | §다른 인코더 계열(reg-count 스윕) | k=0→4, 42.30→70.74 단조증가 |

## 주의
- `eval_v2_seeds3_50k.json`·`deit_compare_report.txt`는 파일명 충돌/명확성을 위해 이름을 조금 바꿔 복사했습니다(원본: `eval_v2/results_50k/eval_v2_seeds3.json`, `deit_compare/results/comparison_report.txt`).
- 이 파일들은 **복사본**입니다. 실험 재현·수정은 원본 위치(`vision_models/register_token_reduction/…`)에서 하세요.
- 모두 단일 seed(핵심 분류표) 또는 3-seed(보조)이며, 논문의 정직성 원칙대로 외부 논문 수치 인용 없이 동일 프로토콜에서 직접 측정한 값입니다.
