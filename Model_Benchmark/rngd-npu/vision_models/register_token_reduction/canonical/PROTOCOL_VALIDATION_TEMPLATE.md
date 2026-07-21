# 정통(train-갤러리) 검증 — main.md 삽입 템플릿 (결과 도착 시 채워서 붙여넣기)

> base(ViT-B) 정통 결과가 `base_results/results_base_canonical_{g0,g1}.txt`에 도착하면, 아래 [TODO]에 실제 수치를 채워 main.md 실험 절 끝(적용범위와한계 앞)에 삽입한다. **수치 확인 전 붙여넣지 말 것.** 순서(Ours>PiToMe 극단, Ours>ToMe)가 val-LOO와 다르면 프레이밍부터 재검토.

## (A) 삽입할 subsection (실험 절, ablation/aux 뒤·적용범위와한계 앞)
```latex
\subsection{프로토콜 검증: 표준 train-갤러리 kNN}
주 결과는 계산 편의상 val leave-one-out kNN을 쓴다(모든 방법 동일). 이 선택이 결론을 왜곡하지
않음을 확인하기 위해, 헤드라인 인코더(DINOv2-ViT-B/14)를 표준 프로토콜(gallery = ImageNet
train 전체 128만, query = val, $k{=}20$)로 재평가한다(Table~\ref{tab:canonical}). 무압축
baseline은 공인값에 부합하는 $[TODO r=0]$이며(문헌 $82.0$), 레지스터 보호의 우위 순서
(Ours $>$ ToMe 전구간, Ours $>$ PiToMe 극단)는 이 프로토콜에서도 유지된다. 즉 상대적 결론은
갤러리 선택에 불변이다.

\begin{table}[t]
\centering
\caption{표준 train-갤러리 kNN(gallery=ImageNet train, query=val, $k{=}20$)에서의 재현.
DINOv2-ViT-B/14. 무압축 $[TODO]$(문헌 $82.0$).}
\label{tab:canonical}
\begin{tabular}{lccc}
\toprule
토큰 축소 & ToMe & PiToMe & \textbf{Ours} \\
\midrule
37\% & [TODO] & [TODO] & [TODO] \\
55\% & [TODO] & [TODO] & [TODO] \\
74\% & [TODO] & [TODO] & [TODO] \\
83\% & [TODO] & [TODO] & [TODO] \\
92\% & [TODO] & [TODO] & \textbf{[TODO]} \\
\bottomrule
\end{tabular}
\end{table}
```

## (B) 결론 문장 플립 (검증 성공 시)
- 현재: "…표준 train-갤러리 프로토콜과 온칩(NPU) 가속은 향후 과제로 남는다…"
- → "표준 train-갤러리 프로토콜에서도 헤드라인 결과를 재현했고(Sec.~\ref{...}), 온칩(NPU) 가속만 향후 과제로 남는다." (S/L 정통은 미완이면 "다른 크기로의 정통 확장"을 향후로.)

## (C) 채우는 법
1. `results_base_canonical_g0.txt`(r=0,8,12,16) + `_g1.txt`(r=0,18,20) 두 파일에서 r=0(baseline)·각 r의 tome/pitome/ours 값을 읽음.
2. comp% 매핑: r=8→37, 12→55, 16→74, 18→83, 20→92.
3. r=0 값이 ~82 근처인지 먼저 확인(양쪽 파일이 교차검증). 순서 유지 확인.
4. 위 표 [TODO] 채우고 main.md·main_en.md 양쪽에 삽입 + 결론 플립. **main_en.md도 함께**(현재 대등).
5. 이 검증이 §aux나 §설정 중 어디가 자연스러운지 판단(현재 계획=실험 절 끝 독립 subsection).

⚠️ S/L 정통까지 있으면 표에 행 추가하거나 별도 표. 없으면 "헤드라인(ViT-B) 검증"으로 명시.
