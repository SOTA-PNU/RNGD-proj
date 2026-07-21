\documentclass[runningheads]{llncs}

% ---------------------------------------------------------------
% ACCV 기본 패키지 (review 모드)
\usepackage[review,year=2026,ID=*****]{accv}
%\usepackage{accv}          % TODO FINAL: camera-ready

\usepackage{accvabbrv}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[accsupp]{axessibility}
\usepackage[pagebackref,breaklinks,colorlinks,citecolor=accvblue]{hyperref}
\usepackage{orcidlink}
\usepackage{kotex}          % NOTE: 한국어 초안. 컴파일 시 kotex(한글) 필요. 최종 투고본은 영어로 전환.
\setlength{\tabcolsep}{9pt}  % 표 열 간격 확대(헤더 항목 구분). 7열 tab:pitome이 넘치면 값을 낮추거나 \resizebox 사용.
\renewcommand{\arraystretch}{1.15}  % 행 높이 살짝 확대(가독성)

\begin{document}

% ---------------------------------------------------------------
% NOTE: 이 파일은 한국어 작업 초안입니다(내용 검토용). ACCV 투고는 영어이며,
%       영어 동기화본은 main_en.md 입니다(2026-07-02 이 파일과 대등 동기화 완료:
%       47인용·표수치·구조 일치). main_en_backup.md 는 2026-07-01 옛 스냅샷.
% 원칙: 본문 주장은 전체 규모(ImageNet val 50k, ADE20k val 전체) 결과만 사용하고,
%       소규모(n=3000) 검증은 "예비 보조검증"으로 명시 구분한다.

\title{레지스터를 병합하지 말라: 레지스터를 가진 비전 인코더의
극단적 토큰 압축을 위한 레지스터 인지 축소}

\titlerunning{Register-Aware Token Reduction}
\author{익명 (ACCV 제출)}
\authorrunning{Anonymous ACCV submission}
\institute{Paper ID *****}
% === CAMERA-READY 저자 (채택 후 활성화; 리뷰본은 위 익명 유지) ===
% \author{Hyunjun Cho\inst{1} \and Joohyoung Cha\inst{1} \and Yongin Kwon\inst{2}}
% \authorrunning{H. Cho et al.}
% \institute{University of Science and Technology (UST), Korea \\ \email{[UST 이메일]}
% \and Pusan National University, Korea \\ \email{[부산대 이메일]}}

\maketitle


\begin{abstract}
DINOv2와 CLIP 같은 파운데이션 비전 인코더는 고정 특징 추출기로 널리 쓰이지만, 한 이미지를 수백 개의 토큰으로 처리해 추론 비용이 크다. 토큰을 병합하거나 가지치는 토큰 축소는 이 비용을 낮추는데, 기존 방법은 유사도나 에너지 같은 일반 신호로 토큰의 중요도를 매긴다. 그런데 이런 인코더 중 일부는 전역 정보를 소수의 레지스터 토큰에 몰아넣는다~\cite{registers}. 우리는 표준 크기-가중 병합이 극단 압축에서 이 레지스터 토큰까지 합쳐 없애 정확도가 크게 떨어짐을 관찰하고, 재학습 없이 레지스터를 지키고 나머지 패치만 공격적으로 병합하는 규칙을 연구한다.

레지스터 4개를 가진 DINOv2-ViT-B/14를 표준 프로토콜로 재면, 레지스터를 보호할 때 ${\sim}92\%$ 축소에서 $77.3\%$를 유지하는 반면 보호하지 않는 ToMe는 $70.0\%$로 떨어져 $+7.3$의 차이를 보인다. 여기서 갤러리는 ImageNet train 전체이고 무압축 정확도는 $80.9$로 공인값 $82$에 가깝다. PiToMe는 같은 병합에 에너지 기준을 얹은 방법인데, 같은 예산에서 함께 재면 두 규칙의 정확도 차가 압축과 함께 커져 온건한 $+0.6$에서 ${\sim}92\%$의 $+6.2$에 이른다. 같은 레지스터 보호를 PiToMe 병합 위에 얹어도 모든 축소율에서 이득이 남아, 이 규칙이 두 병합기 모두에서 성립한다. 규칙은 DINOv2를 넘어 RoPE를 쓰는 DINOv3와 ViT-5로도 이어져, 레지스터를 지키면 극단 압축에서도 정확도가 유지되지만 제거하면 무너진다.

우리는 새 병합 알고리즘이 아니라 레지스터를 압축의 보호 기준으로 삼는 규칙을 제안하며, 이는 PiToMe가 에너지 규칙을 더한 것과 같은 자리다. 활성 양자화를 위해 레지스터를 앞에 붙이는 RegCache~\cite{regcache}나 어텐션을 근사하는 FNA~\cite{fna}와 달리, 우리는 레지스터를 시퀀스 길이 축소의 보호 기준으로 쓴다. ADE20k 선형 프로브 분할에서도 같은 보호가 모든 축소율에서 mIoU를 높이고 이득이 압축과 함께 커지는 반면, 대안 기준들은 무보호에서 거의 벗어나지 못한다. 방법이 결정적이라 평가셋을 부트스트랩해 불확실성을 정량화했고, 이득의 $95\%$ 신뢰구간은 모든 축소율에서 $0$을 배제한다.
  \keywords{토큰 축소 \and Vision Transformer \and 레지스터 토큰 \and 효율적 추론}
\end{abstract}


\section{서론}
\label{sec:intro}

Vision Transformer(ViT)~\cite{vit} 기반 파운데이션 비전 인코더는 분류와 검색, 밀집 예측의 기본 백본으로 자리잡았다. 자기지도로 학습하는 DINOv2~\cite{dinov2}는 마스크 이미지 모델링과 자기증류~\cite{mae,beit,ibot}의 계보를 잇고, 언어지도 대조학습 계열로는 CLIP~\cite{clip}과 SigLIP~\cite{siglip}이 있다. ViT는 한 이미지를 수백 개의 토큰으로 바꿔 처리한다. 대부분은 이미지를 격자로 자른 패치이고, 이미지 전체를 대표해 최종 특징으로 읽히는 클래스 토큰 같은 소수의 특수 토큰이 함께 있다. 자기어텐션 비용이 토큰 수의 제곱에 비례해 커지므로 입력 해상도와 모델이 커질수록 추론이 빠르게 느려진다. 이를 완화하려고 재학습 없이 네트워크 안에서 비슷하거나 덜 중요한 토큰을 병합하거나 가지치는 토큰 축소~\cite{tome,pitome,evit,dynamicvit}가 널리 쓰인다.

토큰 축소의 핵심 질문은 어떤 토큰을 남기고 어떤 토큰을 합칠지이며, 기존 방법은 이를 모델 비의존적 일반 신호로 정한다. ToMe는 특징 유사도를 쓰고, PiToMe는 토큰 간 유사도 그래프에서 각 토큰의 중요도를 재는 스펙트럼 에너지를 쓴다. 그런데 최근 연구는 ViT가 소수의 특별한 토큰에 전역 정보를 몰아넣음을 보인다. 레지스터와 함께 학습하면 이는 별도의 레지스터 토큰에 담기고, 그렇지 않으면 배경 자리의 고노름 이상치 토큰~\cite{massive}에 담긴다. 이 토큰들은 노름이 유독 커서 어텐션이 몰리는 attention sink로 작동하며 전역 문맥을 모은다. 우리는 이런 레지스터 인코더에서 토큰의 정체를 보지 않는 표준 병합이 극단 축소에서 이 소수의 전역 토큰을 평범한 패치와 함께 이르게 합쳐 없애고, 그 결과 정확도가 크게 무너짐을 관찰한다.

이 관찰에서 출발해 우리는 레지스터 인지 토큰 축소를 연구한다. 학습이 필요 없는 간단한 규칙으로, 명시적 레지스터와 클래스 토큰을 보호 집합으로 두어 병합에서 제외하고 나머지 중복된 패치 토큰만 공격적으로 병합한다. 이는 새 병합 알고리즘이 아니라 ToMe 병합 프레임워크 안에서 무엇을 지킬지 정하는 보호 규칙이며, PiToMe가 같은 자리에 에너지 규칙을 더한 것과 나란하다. 본 논문은 명시적 레지스터를 가진 인코더에 집중한다.

이 규칙이 얼마나 견고한지 여러 방식으로 확인한다. DINOv2-ViT-B/14를 표준 train-갤러리 kNN으로 재면 ${\sim}92\%$ 축소에서 무보호 ToMe 대비 $+7.3$을 얻고 이득은 압축과 함께 커진다. 같은 프레임워크의 다른 규칙인 공식 PiToMe와 같은 예산에서 함께 재도 두 규칙의 정확도 차가 압축과 함께 벌어지며, 같은 보호를 PiToMe 병합 위에 얹어도 이득이 유지된다. 같은 개수를 다른 기준으로 보호하는 실험은 이득이 토큰 수가 아니라 레지스터에서 옴을 보인다. 규칙은 DINOv2를 넘어 DINOv3와 ViT-5, 밀집 예측으로도 이어진다.


\section{관련 연구}
\label{sec:related}

\paragraph{토큰 축소.} ToMe는 학습 없이 크기 가중 이분 소프트 매칭으로 유사한
토큰을 병합하며, 이후 토큰 축소 연구가 딛고 서는 프레임워크를 세웠다. 이 프레임워크에서 남는 핵심 문제는 \emph{어떤 토큰을 합치고 남길지}이며,
PiToMe는 그 이분 소프트 매칭을 그대로 두고 스펙트럼 에너지 기준을 더하고,
DynamicViT·EViT는 학습된 마스크를 쓴다. 이 밖에도 적응적 토큰 샘플링~\cite{ats}, 적응적
halting~\cite{avit}, 슬로우-패스트 토큰 진화~\cite{evovit}, 학습형 압축률~\cite{diffrate},
프루닝과 병합의 통합~\cite{tofu,ltmp}, 학습형 병합·풀링~\cite{patchmerger,tokenpooling},
극소수 학습 토큰~\cite{tokenlearner} 등 다양한 변형이 연구되었으며, 이들의 체계적 비교는
Haurum 등~\cite{haurum}에 정리돼 있다. 그러나 이들의 규칙은 모두 모델 비의존적 일반 신호(유사도·에너지)를 쓰며 레지스터/이상치
토큰 구조를 이용하지 않는다. 우리는 같은 병합 프레임워크 안에서, 모델 구조가 알려주는 \emph{레지스터}를 보호 기준으로 삼는 새 규칙을 제안한다. 이는 PiToMe가 같은 프레임워크에 에너지 규칙을 더한 것과 같은 자리다.

\paragraph{레지스터와 이상치 토큰.} Darcet 등은 ViT가 정보가 적은 배경
위치에 고노름 이상치 토큰을 만들어 내부 계산용으로 전용함을 발견하고, 이를 흡수할 전용
레지스터 토큰을 추가해 특징·어텐션 지도를 매끄럽게 하고 밀집 예측을 개선했다. Sun 등은 이를 ``massive activations''으로 특징짓는다. 이 연구들의 목적은 효율이 아니라 표현 품질이며, 우리는 이
레지스터를 \emph{압축의 보호 기준}으로 재해석한다. 같은 고노름 현상은 대형 언어모델에서도 소수
채널·토큰에 집중돼 양자화를 어렵게 하고~\cite{llmint8,smoothquant}, 특정 토큰에 어텐션이
쏠리는 attention sink로 이어지며~\cite{streamingllm,attnsinkemerge,quantizabletransformers}, 이
성분을 함부로 제거하면 성능이 붕괴한다~\cite{bertbusters,outlierfreq}. 비전에서도 이 아티팩트가 밀집 예측을
저해하며~\cite{denoisingvit}, 학습 없이 테스트시 레지스터를 주입해 그 행동을 재현할 수
있다~\cite{testtimeregisters}.

\paragraph{이상치 토큰을 효율에 활용.} 우리와 가장 가까운 것으로, RegCache는 활성
\emph{양자화}를 위해 레지스터를 앞에 붙여 이상치를 막고, FNA는 거대/이상치
토큰을 랜드마크로 삼아 어텐션을 선형 시간으로 \emph{근사}한다. 둘 다 모든 토큰을 유지한다.
우리는 대신 레지스터를 보호 기준으로 삼아 \emph{시퀀스 길이를 줄인다}. 넓게 보면 우리의 시퀀스 길이 축소는
어텐션 연산 자체를 근사·선형화하는 효율 축~\cite{flashattention,linformer,performer,reformer,nystromformer}과
상보적·직교적이다.



\section{레지스터 인지 토큰 축소}
\label{sec:method}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_method.pdf}
\caption{레지스터 인지 토큰 축소. 각 블록은 어텐션, 병합, MLP를 차례로 적용하며, 패치 토큰은 크기 가중 이분 소프트 매칭으로 한 번에 $r$쌍씩 병합되고, 보호 집합은 병합에서 제외되어 그대로 통과한다. 위: ToMe는 클래스 토큰만 보호해 극단 압축에서 레지스터가 이르게 합쳐져 사라진다. 아래: 우리 방법은 보호 집합을 레지스터까지 넓혀 마지막 블록까지 레지스터를 보존한다.}
\label{fig:method}
\end{figure}

\paragraph{사전지식: 보호 집합을 둔 병합.}
ViT는 토큰 시퀀스 $X=[x_1,\dots,x_T]$를 처리하며, 앞쪽 $p$개 토큰이 전역 토큰이다. 곧 클래스 토큰과 $p{-}1$개의 레지스터다. ToMe~\cite{tome}가 도입한 토큰 병합은 매 블록의 어텐션과 MLP 사이에 병합 단계를 끼워 넣는다. 토큰을 번갈아 두 묶음으로 나누고, 한 묶음의 각 토큰을 어텐션 키의 코사인 유사도로 다른 묶음에서 가장 비슷한 짝과 매칭한 뒤, 점수가 높은 $r$쌍을 크기 가중 평균으로 합친다. 합쳐진 토큰은 자신이 대표하는 원래 토큰 수인 \emph{크기}를 지니며, 어텐션 로짓이 크기에 맞춰 재조정된다(\emph{비례 어텐션}). 이 프레임워크에는 매칭에서 제외되는 보호 집합 $\mathcal{P}$가 이미 들어 있고, ToMe에서는 $\mathcal{P}=\{\text{CLS}\}$다. PiToMe~\cite{pitome}는 이 구조를 그대로 두고 쌍 선택 기준만 스펙트럼 에너지 점수로 바꾸며, 보호 집합은 건드리지 않는다.

\paragraph{보호 규칙.}
우리는 보호 집합을 프리픽스 전체로 넓힌다.
\begin{equation}
\mathcal{P}=\{\text{CLS}\}\cup\{\text{레지스터}\},
\end{equation}
그러면 레지스터는 모든 병합 단계를 그대로 통과하고, 나머지 패치 토큰만 이전과 같은 강도로 병합된다(Fig.~\ref{fig:method}). 이 규칙은 학습이 필요 없고 추가 계산도 없다. 매칭도 평균도 바꾸지 않으므로 병합기와 무관하며, ToMe와 PiToMe에 똑같이 그대로 적용된다.

\paragraph{레지스터를 보호해야 하는 이유.}
레지스터는 학습되는 이미지 무관 토큰으로, 국소 정보가 아니라 전역 정보를 응축한다. 이 토큰에 대한 선형 프로브는 클래스 라벨은 잘 복원하지만 공간 위치는 잘 복원하지 못하며~\cite{registers}, 큰 ViT에서는 전역 표현이 사실상 레지스터 토큰에 지배된다~\cite{clsregdecouple}. 레지스터는 노름이 유난히 커 어텐션 sink로 작동하며 클래스 토큰 어텐션의 상당 부분을 빨아들인다(Fig.~\ref{fig:attention}). 반면 병합은 토큰의 정체를 보지 않는다. 매칭은 키 유사도와 크기만 볼 뿐 레지스터를 여느 토큰과 똑같이 다룬다. 온건한 압축에서는 중복 패치만으로 병합 예산이 채워져 이것이 거의 문제되지 않는다. 그러나 극단 압축에서는 레지스터가 매칭 후보가 되기 쉽고, 레지스터가 패치 무리에 평균으로 섞이면 그 내용이 희석되어 복원되지 않는다. 최종 표현을 클래스 토큰만으로 읽더라도 레지스터가 담고 있던 전역 정보는 사라진다. 우리는 이 설명을 직접 검증한다. 보호하지 않는 스케줄을 추적하면 대부분의 레지스터가 네트워크 전반부에서 흡수되며, 같은 수의 토큰을 다른 어떤 기준으로 보호해도 정확도가 회복되지 않는다.

\paragraph{구현.}
우리는 이 규칙을 선행 방법과 같은 정식 하네스 안에서 구현하며, 바꾸는 것은 보호 집합뿐이라 측정되는 차이는 오롯이 보호에서 온다. 얇은 어댑터가 각 공식 모델을 가중치 변경 없이 감싸 무압축 forward를 정확히 재현하며, $r{=}0$에서 기준 특징과 코사인 유사도가 $1.000$이라 우리가 보고하는 모든 변화를 재구현이 아니라 축소 탓으로 돌릴 수 있다. 위치는 병합 시점에 처리한다. 두 패치가 병합되면 살아남는 토큰이 자신을 흡수한 패치의 격자 위치를 물려받아, 모든 토큰이 병합 뒤에도 정의된 위치를 유지한다. 이는 회전 위치 임베딩 RoPE~\cite{rope}에서 중요하다. RoPE는 위치 벡터를 더하는 대신 각 토큰의 질의와 키를 그 위치에 비례한 각도만큼 회전시키고 이 회전을 매 블록 다시 걸어 상대 위치를 표현한다. DINOv3와 ViT-5는 레지스터와 패치를 서로 다른 위치 공간에 두므로 레지스터를 패치에 평균으로 섞으면 합쳐진 토큰의 회전이 정의되지 않는데, 레지스터가 보호되고 병합된 패치는 실재 격자 위치를 물려받으므로 우리 축소는 두 공간을 섞지 않고 RoPE에서도 유효하다.

\paragraph{적용 방식과 범위.}
우리는 이 규칙을 ToMe와 PiToMe 위에 똑같이 적용하며, 그들의 스케줄도 하이퍼파라미터도 바꾸지 않는다. 추가로 남기는 소수의 레지스터 때문에 무보호 baseline 대비 FLOP 차이는 무시할 만하다($<0.1\%$). 밀집 예측에서는 병합된 토큰을 다시 풀어(un-merge) 그 구성 토큰들의 위치로 되돌려, 읽어내기 전에 전체 패치 격자를 복원한다. 이 규칙은 명시적 레지스터를 전제한다. 레지스터가 없는 인코더에서는 고노름 상위 토큰을 보호하는 것이 자연스러운 유사물이지만, 그런 인코더는 이미 병합에 강건한 것으로 드러나므로, 이를 방법이 아니라 적용 범위의 한계로 밝힌다.


\section{실험}
\label{sec:exp}

\subsection{실험 설정}
\textbf{데이터셋과 지표.} 레지스터 4개를 가진 고정된 DINOv2-ViT-B/14를 ImageNet-1k~\cite{imagenet,ilsvrc}에서 kNN 분류~\cite{dino,instdisc}로 평가한다. kNN은 질의 이미지의 특징을 라벨된 참조 집합인 갤러리와 견줘 가장 가까운 $k{=}20$개의 라벨을 다수결하는 학습 없는 방법이다. 주 프로토콜은 표준 DINOv2 kNN 그대로 갤러리=학습셋 전체 $1{,}281{,}167$장, 질의=검증셋 $50{,}000$장 · $224{\times}224$ 입력이며, 무압축 특징의 kNN top-1이 $80.9\%$로 공인값 ${\sim}82$에 근접해 절대 수치가 표준과 정합함을 확인한다. 추가로 val $50$k 자체를 갤러리이자 질의로 쓰는 방식도 있다. 이때 각 이미지는 자기 자신을 이웃에서 빼고 분류하며, 이를 val leave-one-out이라 한다. 여기서도 재측정해 절대 수치는 낮아도 방법 간 \emph{상대 격차}가 갤러리 선택에 불변임을 뒤에서 확인한다.

\textbf{구현 세부.} 토큰 축소는 블록마다 $r$개를 크기 가중 ToMe 방식으로 병합하며,\documentclass[runningheads]{llncs}

% ---------------------------------------------------------------
% Include basic ACCV package
 
% TODO REVIEW: Insert your submission number below by replacing '*****'
% TODO FINAL: Comment out the following line for the camera-ready version
\usepackage[review,year=2026,ID=794]{accv}
% TODO FINAL: Un-comment the following line for the camera-ready version
%\usepackage{accv}

% OPTIONAL: Un-comment the following line for a version which is easier to read
% on small portrait-orientation screens (e.g., mobile phones, or beside other windows)
%\usepackage[mobile]{accv}


% ---------------------------------------------------------------
% Other packages

% Commonly used abbreviations (\eg, \ie, \etc, \cf, \etal, etc.)
\usepackage{accvabbrv}

% Include other packages here, before hyperref.
\usepackage{graphicx}
\usepackage{booktabs}

% The "axessiblity" package can be found at: https://ctan.org/pkg/axessibility?lang=en
\usepackage[accsupp]{axessibility}  % Improves PDF readability for those with disabilities.


% ---------------------------------------------------------------
% Hyperref package

% It is strongly recommended to use hyperref, especially for the review version.
% Please disable hyperref *only* if you encounter grave issues.
% hyperref with option pagebackref eases the reviewers' job, but should be disabled for the final version.
%
% If you comment hyperref and then uncomment it, you should delete
% main.aux before re-running LaTeX.
% (Or just hit 'q' on the first LaTeX run, let it finish, and you
%  should be clear).

% TODO FINAL: Comment out the following line for the camera-ready version
\usepackage[pagebackref,breaklinks,colorlinks,citecolor=accvblue]{hyperref}
% TODO FINAL: Un-comment the following line for the camera-ready version
%\usepackage{hyperref}

% Support for ORCID icon
\usepackage{orcidlink}


\begin{document}

\title{Do Not Merge the Registers: Register-Aware Reduction for
Extreme Token Compression in Vision Encoders with Registers}

\titlerunning{Register-Aware Token Reduction}

% TODO FINAL: Replace with your author list. 
% Include the authors' OCRID for the camera-ready version, if at all possible.
\author{First Author\inst{1}\orcidlink{0000-1111-2222-3333} \and
Second Author\inst{2,3}\orcidlink{1111-2222-3333-4444} \and
Third Author\inst{3}\orcidlink{2222--3333-4444-5555}}

% TODO FINAL: Replace with an abbreviated list of authors.
\authorrunning{F.~Author et al.}
% First names are abbreviated in the running head.
% If there are more than two authors, 'et al.' is used.

% TODO FINAL: Replace with your institution list.
\institute{Princeton University, Princeton NJ 08544, USA \and
Springer Heidelberg, Tiergartenstr.~17, 69121 Heidelberg, Germany
\email{lncs@springer.com}\\
\url{http://www.springer.com/gp/computer-science/lncs} \and
ABC Institute, Rupert-Karls-University Heidelberg, Heidelberg, Germany\\
\email{\{abc,lncs\}@uni-heidelberg.de}}
``ToMe''는 클래스 토큰만, ``Ours''는 레지스터 토큰까지 보호한다. 본 논문에서 축소율(\%)은
\emph{토큰 수} 기준이며, 병합이 점진적이라 실제 FLOP 절감은 이보다 작아, 토큰 ${\sim}92\%$ 축소는 FLOP 기준 ${\sim}43\%$ 절감에 해당한다. 분류 실험 Table~\ref{tab:main}, \ref{tab:pitome}, \ref{tab:generality}, \ref{tab:ablation}은 모두 표준 train-갤러리에서 전체 규모, 단일 seed로 측정하며, 하네스는 선행연구 그대로 비례 어텐션과 어텐션 키 유사도, 어텐션과 MLP 사이 병합을 쓴다. 다중 인코더와 선형 프로브, 동적 기준 결과는 3-seed 평균이며 seed 표준편차는 작다. 무압축 상한을 포함해 본 논문의 모든 정확도 · FLOP 수치는 \emph{동일 프로토콜}에서 우리가 직접 측정한 값이며, 외부 논문 수치를 인용하지 않는다. 검색 mAP는 검증셋 내부 자기검색 지표라 val에서 보고한다.

\subsection{주 결과 (표준 train-갤러리 kNN)}
Table~\ref{tab:main}은 표준 train-갤러리 kNN에서의
정확도를 보고한다. 레지스터 보호는 측정한 모든 축소율에서 정확도를
높이며, 그 이득은 압축이 강할수록 커져 $37\%$의 $+0.9$에서 ${\sim}92\%$의 $+7.3$에
이른다. 가장 공격적인 설정에서 우리 방법은 $77.3\%$를 유지해 무압축 $80.9$보다 $3.6$%p 낮은 반면, 무보호 베이스라인은 $10.9$%p 낮은 $70.0\%$로 떨어진다. 축소율별 세부 값은 단일
seed이라 유의성 검정을 하지 않으며, 레지스터 개수 스윕의 부트스트랩 신뢰구간도 함께 보고한다.

\begin{table}[t]
\centering
\caption{표준 train-갤러리 kNN(gallery = ImageNet train $1.28$M, query = val
$50{,}000$, $k{=}20$)에서의 top-1 정확도(\%). DINOv2-ViT-B/14 레지스터 4개, 선행연구 그대로의 정식 ToMe. 무압축 모델:
$80.87$(공인 ${\sim}82$). ``ToMe''는 클래스 토큰만, ``Ours''는 레지스터 토큰까지 보호.
극단 설정에서 Ours는 보호된 레지스터만큼 ToMe보다 토큰을 약간 더 유지한다(단일 seed).}
\label{tab:main}
\begin{tabular}{lccc}
\toprule
토큰 축소 & ToMe & \textbf{Ours} & $\Delta$ \\
\midrule
37\% & 79.64 & \textbf{80.53} & $+0.89$ \\
55\% & 78.52 & \textbf{80.15} & $+1.63$ \\
74\% & 75.99 & \textbf{79.41} & $+3.42$ \\
83\% & 73.79 & \textbf{78.67} & $+4.88$ \\
92\% & 70.00 & \textbf{77.28} & $+7.28$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{PiToMe와 같은 예산에서 재본 다른 선택 규칙}
주 결과의 ToMe는 유사도만 쓰는 기본 병합이다. PiToMe는 같은 이분 소프트 매칭 위에 에너지 기준을 얹은 방법이고, 우리 레지스터 보호는 같은 프레임워크에 얹은 또 다른 선택 규칙이다. 두 규칙이 \emph{같은 예산}에서 어떻게 측정되는지 같은 하네스로 함께 잰다. 공식
PiToMe의 에너지 selection(층별 마진 $m{=}0.75{-}0.75\,\ell/L$, 고에너지
병합 · 저에너지 보호; 앞 절반 블록은 공식 \texttt{pitome\_bsm}, 뒤 절반은 에너지 경로)을 우리 공통
하네스에 소스대로 이식하고, ToMe · PiToMe · Ours를 같은 모델 ·
같은 프로토콜 · 같은 축소율에서 재측정한다(Table~\ref{tab:pitome}). 세 방법 모두 선행연구 그대로의 정식 하네스 위에서 돌아 유일한 차이가 선택·보호 규칙이므로 비교는 통제된다.
같은 하네스에 레지스터 보호를 더하면 격차가 압축과 함께 커져 온건한 압축의
$+0.6$에서 ${\sim}92\%$의 $+6.2$에 이른다. 계산량을 같은 기준으로 비교하려고 축소를 FLOP 기준으로도 Table~\ref{tab:pitome}에 함께 보고한다. 병합이 점진적이어서 앞 블록은 여전히 많은 토큰을 처리하므로, \emph{토큰 수} 기준 축소는 실제 FLOP 절감을 과대평가한다. 토큰 ${\sim}92\%$ 축소는 FLOP 기준 ${\sim}43\%$ 절감이다. 우리 sweep은 FLOP $17$--$43\%$ 절감 범위로, PiToMe가 보고한 $40$--$60\%$와 동일 FLOP 기준에서 비교되며, 우리는 더 낮은 예산 영역을 다룬다. 같은 축소율에서 처리량은 세 방법이 비슷해 ${\sim}92\%$에서 모두 ${\sim}570$ im/s대다. 즉 레지스터 보호는 속도를 깎지 않는다. 요컨대 \textbf{우리는 더 나은 병합 알고리즘을
주장하는 것이 아니라, 레지스터를 압축의 명시적 보호 기준으로 삼는 규칙을 제안한다}. 이는 PiToMe의 에너지 규칙처럼 표준 토큰 병합 프레임워크 안에서 동작하며, 속도 손해 없이 극단 압축에서도 정확도 하락을 작게 유지한다.

\begin{table}[t]
\centering
\caption{공식 PiToMe와의 같은 예산 직접 비교. DINOv2-ViT-B/14 레지스터 4개, 표준
train-갤러리 kNN($k{=}20$), 무압축 $80.87$. 세 방법 모두 우리 하네스에서 재측정.
각 kNN 열에서 최고는 볼드, 차선은 밑줄.
``FLOP 절감''은 점진적 병합의 실제 계산 절감으로 토큰 수 축소보다 작다. ``GFLOPs''는 이미지 1장당 계산량으로 무압축은 $23.5$이며, ToMe·PiToMe와 같은 표준 관행인 \texttt{fvcore}의 곱셈-누산 셈을 따른다.}
\label{tab:pitome}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lllcccc}
\toprule
토큰 축소 & FLOP 절감 & GFLOPs\,$\downarrow$ & ToMe & PiToMe & \textbf{Ours} & $\Delta$(O$-$P) \\
\midrule
37\% & 17\% & 19.4 & 79.64 & \underline{79.89} & \textbf{80.53} & $+0.64$ \\
55\% & 26\% & 17.4 & 78.52 & \underline{79.16} & \textbf{80.15} & $+0.99$ \\
74\% & 35\% & 15.4 & 75.99 & \underline{76.94} & \textbf{79.41} & $+2.47$ \\
83\% & 39\% & 14.4 & 73.79 & \underline{74.73} & \textbf{78.67} & $+3.94$ \\
92\% & 43\% & 13.4 & 70.00 & \underline{71.08} & \textbf{77.28} & $+6.20$ \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{왜 레지스터를 보호하는가?}
레지스터가 아예 없는 DINOv2(prefix${=}1$)에서는 Ours와 ToMe가 같은 계산이 되어 차이가 사라지고, 재실행 노이즈는 $0.05$%p 이하다. 이는 이득이 레지스터 없이는 생기지 않음을 보이지만, 이득이 정말 레지스터에서 오는지는 다음 실험이 가린다. 병합을 고정하고 클래스 토큰 외에 같은 개수를 무엇으로 보호할지만 바꾸면 레지스터만 도움이 된다. ${\sim}92\%$ 축소에서 같은 개수를 무작위로 보호하면 $69.5\%$, 평균 유사도 기준으로 $70.2\%$, 고노름으로 $70.2\%$가 되어 모두 무보호 베이스라인 $70.0\%$ 근처에 머문다. 반면 레지스터를 보호하면 $77.28\%$로 이 정적 기준들보다 약 $+7$%p 높다. 비레지스터 기준들은 베이스라인 근처에서 오르내려 계통적 이득이 없으므로 그들 사이의 순서는 주장하지 않는다.

\begin{table}[t]
\centering
\caption{보호 기준을 바꾼 비교. 병합 방식은 그대로 두고 클래스 토큰 외에 같은 개수를 무엇으로 보호할지만 바꾼다. 레지스터를 보호할 때만 무보호 베이스라인을 크게 넘어서고, 무작위나 평균 유사도, 고노름으로 보호하면 베이스라인 근처에 머문다. 에너지는 평균 유사도로 흉내 낸 프록시이며 공식 PiToMe와 다르다. DINOv2-reg, 표준 train-갤러리 kNN, $k{=}20$, 단일 seed.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
축소 & ToMe & \textbf{Ours} & 무작위 & 에너지 & 고노름 \\
\midrule
37\% & 79.64 & \textbf{80.53} & 79.71 & 79.67 & 79.67 \\
55\% & 78.52 & \textbf{80.15} & 78.45 & 78.64 & 78.56 \\
74\% & 75.99 & \textbf{79.41} & 76.05 & 76.17 & 76.29 \\
83\% & 73.79 & \textbf{78.67} & 73.83 & 74.02 & 74.11 \\
92\% & 70.00 & \textbf{77.28} & 69.53 & 70.19 & 70.20 \\
\bottomrule
\end{tabular}
\end{table}

한 가지 중요한 단서를 붙인다. Table~\ref{tab:ablation}의 무작위/에너지/고노름 집합은
입력층 임베딩에서 한 번 골라 고정한 것인데, 고노름 이상치 토큰은 네트워크 중간에서 생긴다.
전체 $50{,}000$ val에서 base 모델을 kNN으로 잰 통제 하네스 보조 점검에서, 고노름이나 에너지 집합을 매 블록마다
\emph{동적으로} 다시 고르면 그 베이스라인들이 강해진다. ${\sim}91\%$ 축소에서 최선의 동적
기준은 $67.4\%$에 이르는 반면, 정적 기준과 무보호는 ${\sim}64\%$, 레지스터 보호는
$71.9\%$이다(이 통제-하네스 절대값은 정식 Table~\ref{tab:ablation} 수치와 직접 비교되지
않으며, 핵심은 같은 하네스 안의 격차다). 즉 동적 재선택이 격차의 약 절반을 메우지만 레지스터 보호가 여전히
${\sim}+4.4$로 앞선다. 따라서 ``레지스터만 효과''라는 읽기는
Table~\ref{tab:ablation}의 \emph{정적} 기준에 엄밀히 성립하고, 동적으로 재선택한 기준에
대해서도 레지스터 보호가 극단 압축에서 여전히 앞선다.

\subsection{심화 분석}
레지스터 보호가 이득을 주고 그 이득이 \emph{레지스터}에서 옴을 확인했으니, 이제 그 원인을 병합과 어텐션에서 직접 관찰하고, 이 이득이 추가 계산 없이 얻어짐을 확인한다.

\textbf{메커니즘: 베이스라인은 레지스터를 이르게 병합한다.} 무보호 베이스라인에서 병합을 직접 추적했다. DINOv2-reg에서 $74\%$ 축소에 해당하는 $r{=}16$으로 이미지
$16$장에 걸쳐 레지스터 토큰의 $94\%$가 다른 토큰으로 병합되며, 첫 병합 블록은 평균 $12$개 중 ${\sim}3.6$번째이며 빠르면 블록 $0$이고, 최종 층까지 살아남는 것은 $6\%$뿐이다.
이들을 보호하는 Ours는 구성상 모든 레지스터를 유지한다. 이는 이득이 간접 효과가 아니라
레지스터 토큰의 이른 파괴를 막는 데서 온다는 해석을 뒷받침한다. 다만 작은 이미지 표본과 단일
$r$에서 측정한 것이다. 같은 현상이 어텐션에서도 Fig.~\ref{fig:attention}에 드러난다. 무압축 모델에서 클래스 토큰 어텐션의 상당 부분이 소수의 레지스터에 쏠려 attention sink로 작동하는 반면, 무보호 병합은 이
레지스터를 패치로 합쳐 그 어텐션 구조를 흩뜨린다.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_attention.pdf}
\caption{클래스 토큰이 각 패치에 주는 어텐션이 몰리는 정도를 색으로 나타낸 것으로, 빨강일수록 많이, 파랑일수록 적게 몰린다. 네 패널은 같은 색 스케일을 쓴다. 학습된 DINOv2-reg에서는 소수의 고노름 레지스터가 어텐션을 빨아들이는 sink로 작동한다. 극단 압축 ${\sim}92\%$에서 클래스 토큰만 지키는 ToMe와 PiToMe는 레지스터를 모두 병합해 이 몰림 구조가 흐트러지는 반면, 레지스터를 보호하는 Ours는 무압축과 비슷한 몰림 구조를 유지한다. 마지막 블록의 단일 이미지 정성 예시다.}
\label{fig:attention}
\end{figure}

\textbf{FLOP 절감과 GPU 처리량.} 토큰 축소는 우리 범위에서 백본 FLOP을 $17$--$43\%$ 낮춘다. ${\sim}92\%$ 축소에서 $43\%$이며, 토큰이 점진적으로 제거되고 투영과 MLP FLOP이 지배적이라 감소는 준선형이다. Ours와 무보호 베이스라인의 FLOP은 $<0.1\%$ 차이로 사실상 같아, 추가로 유지되는 레지스터는 무시할 수준이고 정확도 이득을 추가 계산으로 산 것이 아니다. Table~\ref{tab:throughput}에서 보듯 GPU 처리량은 세 방법이 거의 겹치며 압축과 함께 올라 ${\sim}92\%$서 무압축 대비 ${\sim}1.6\times$가 된다.

\begin{table}[t]
\centering
\caption{토큰 축소율에 따른 GPU 처리량(im/s), DINOv2-ViT-B/14. 세 방법이 거의 겹쳐 레지스터 보호가 속도를 깎지 않음을 보인다.}
\label{tab:throughput}
\begin{tabular}{lccc}
\toprule
토큰 축소 & ToMe & PiToMe & \textbf{Ours} \\
\midrule
0\%  & 355 & 351 & 350 \\
37\% & 405 & 396 & 406 \\
55\% & 451 & 440 & 451 \\
74\% & 503 & 493 & 504 \\
83\% & 543 & 531 & 544 \\
92\% & 574 & 562 & 575 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{두 병합기에서 성립하는 보호 규칙}
지금까지 레지스터 보호를 ToMe 병합 위의 keep-rule로 보였다. 이것이 ToMe에만 국한된 효과인지,
아니면 다른 병합기에서도 성립하는지 확인하기 위해, \emph{같은} 레지스터 보호를 공식 PiToMe 병합 위에도 그대로 얹어 PiToMe$+$reg를 만든다. Table~\ref{tab:generality}에서 레지스터를
보호하면 PiToMe 병합에서도 모든 축소율에서 정확도가 오르며(reg 이득 ${>}0$, ${\sim}92\%$서
$+5.1$), 그 이득은 압축과 함께 커진다. 즉 레지스터 보호는 ToMe에만 국한된 효과가 아니라,
우리가 시험한 두 병합기 ToMe와 PiToMe 모두에서 성립하는 keep-rule이다. 이는 PiToMe가 에너지 규칙을 ToMe
프레임워크에 더한 것과 \emph{나란히} 우리는 레지스터 keep-rule을 더한다는 우리 포지셔닝과
정합하며, 두 규칙이 서로 배타적이지 않고 함께 쓰일 수 있음을 보인다.

\begin{table}[t]
\centering
\caption{레지스터 보호를 공식 PiToMe 병합 위에 얹은 PiToMe$+$reg.
DINOv2-ViT-B/14 레지스터 4개, 표준 train-갤러리 kNN($k{=}20$), 무압축 $80.87$. ``reg
이득''$=$ PiToMe$+$reg $-$ PiToMe로, 레지스터 보호가 PiToMe 병합 위에서 주는 추가
이득으로 모든 축소율에서 $>0$이다.}
\label{tab:generality}
\begin{tabular}{lccc}
\toprule
토큰 축소 & PiToMe & \textbf{PiToMe$+$reg} & reg 이득 \\
\midrule
37\% & 79.89 & \textbf{80.37} & $+0.48$ \\
55\% & 79.16 & \textbf{80.11} & $+0.94$ \\
74\% & 76.94 & \textbf{79.06} & $+2.13$ \\
83\% & 74.73 & \textbf{78.13} & $+3.40$ \\
92\% & 71.08 & \textbf{76.15} & $+5.07$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{DINOv3와 ViT-5로의 확장}
지금까지의 실험은 DINOv2에 대한 것이다. 레지스터 보호가 이 모델에만 특유한지 다른 레지스터 인코더에도 성립하는지 보기 위해, 공식 소스에서 그대로 가져온 DINOv3-S+/B~\cite{dinov3}와 ViT-5~\cite{vit5}에서 같은 표준 train-갤러리 kNN 프로토콜로 실험한다. 세 모델 모두 레지스터 $4$개를 쓰지만 DINOv2와 달리 회전 위치 임베딩 RoPE를 쓴다. RoPE는 레지스터와 패치를 서로 다른 위치 공간에 두는데, 우리 축소는 병합 뒤에도 모든 토큰의 위치를 정의된 채로 유지해 두 공간을 섞지 않으므로, 정확도 하락은 위치 오정렬이 아니라 레지스터 손실을 반영한다. 그런 다음 레지스터 보호 축소인 Ours를 레지스터를 제거한 같은 백본과 비교한다. 이 no-reg 백본은 RoPE에서 레지스터의 기여만 안전하게 격리하는 baseline이다.

Table~\ref{tab:extra}에서 세 모델 모두 레지스터를 보호하는 Ours는 극단 압축에서도 정확도가 크게 유지되는 반면, 레지스터를 제거하면 무너진다. 특히 DINOv3는 레지스터 의존이 커서 이를 제거하면 이미 중간 압축에서 무작위 수준으로 붕괴한다. no-reg 정확도는 DINOv3-S+가 ${\sim}48\%$서 $19.6$, ${\sim}95\%$서 $7.9$이고 DINOv3-B는 각각 $14.8$과 $5.6$이다. ViT-5는 좀 더 완만해 $71.2$에서 $61.4$로 하락한다. ${\sim}97\%$까지 밀어붙여 패치가 한 개 이하로 남으면 Ours도 하락해 DINOv3-B는 $54.8$, ViT-5는 $70.7$이 되지만, 레지스터 없는 baseline의 $1.8$과 $42.3$과는 여전히 큰 격차다. 이 극단에서 보호하는 레지스터 수를 $0$에서 $4$로 늘리면 정확도가 DINOv3-B는 $1.8$에서 $54.8$로, ViT-5는 $42.3$에서 $70.7$로 단조 증가해, 이득이 레지스터 보호에서 옴을 재확인한다. 요약하면 레지스터를 보호의 사전지식으로 삼는 규칙은 DINOv2에 국한되지 않고 RoPE 기반 레지스터 인코더로도 이어진다.
(단일 seed. baseline이 ``레지스터 제거''라 DINOv2의 ToMe 베이스라인과 직접 대응하지는 않으며, 여기서는
레지스터 기여의 크기를 격리해 보인다.)

\begin{table}[t]
\centering
\caption{다른 인코더 계열로의 확장. 표준 train-갤러리 kNN top-1. Ours는 레지스터를 보호한 뒤 패치만 병합한 것이고, no-reg는 레지스터를 제거한 같은 백본이다. DINOv3-S+/B와 ViT-5는 RoPE를 쓰며 patch16이라 블록당 $r{=}8/12/16$이 각각 ${\sim}48/72/95\%$ 축소에 해당한다. DINOv2-S/B는 patch14라 축소 지점이 ${\sim}55/74/92\%$로 조금 다르고, 그 no-reg는 레지스터 없이 학습된 별도 DINOv2 모델이다. 우리 어댑터는 $r{=}0$에서 공식 forward와 코사인 유사도가 $1.000$이다.}
\label{tab:extra}
\begin{tabular}{llcccc}
\toprule
모델 & 방법 & 무압축 & ${\sim}48\%$ & ${\sim}72\%$ & ${\sim}95\%$ \\
\midrule
DINOv2-B  & \textbf{Ours} & 80.87 & \textbf{80.15} & \textbf{79.41} & \textbf{77.28} \\
          & no-reg        & 75.85 & 74.57          & 73.73          & 71.68          \\
\midrule
DINOv2-S  & \textbf{Ours} & 77.41 & \textbf{75.80} & \textbf{74.23} & \textbf{69.85} \\
\midrule
DINOv3-S+ & \textbf{Ours} & 77.94 & \textbf{77.37} & \textbf{75.91} & \textbf{70.32} \\
          & no-reg        & ---   & 19.56          & 14.16          & 7.89           \\
\midrule
DINOv3-B  & \textbf{Ours} & 81.63 & \textbf{81.16} & \textbf{80.04} & \textbf{75.09} \\
          & no-reg        & ---   & 14.75          & 10.95          & 5.60           \\
\midrule
ViT-5-B   & \textbf{Ours} & 82.40 & \textbf{81.75} & \textbf{80.84} & \textbf{78.77} \\
          & no-reg        & ---   & 71.17          & 68.36          & 61.43          \\
\bottomrule
\end{tabular}
\end{table}

\subsection{밀집 예측 (ADE20k 분할)}
지금까지는 분류였고, 이제 밀집 예측으로 옮긴다. 밀집 예측은 이미지 하나에 라벨 하나를 매기는 분류와 달리 각 패치나 픽셀마다 클래스를 매기는 과제이며, 분할이 대표적이다. 위치별 정보가 중요하므로 전역 정보를 담은 레지스터를 지키는 것이 분류보다 더 중요할 것으로 예상된다. 품질은 mIoU로 잰다. mIoU는 예측한 영역과 정답 영역이 겹치는 정도를 클래스마다 교집합 나누기 합집합으로 구해 평균한 값으로, 높을수록 좋다.

평가 방법은 이렇다. 고정된 전체 특징 위에 선형 분할 프로브를 학습하고, 병합했던 토큰을 패치 격자로 되돌린 뒤 토큰 축소 하에서 mIoU를 잰다. 백본과 우리 기법은 학습하지 않는다. mIoU를 내려면 특징을 픽셀 클래스로 바꾸는 작은 읽기 층이 반드시 필요한데, 우리는 이 선형 head 하나를 비압축 특징으로 한 번만 학습해 모든 방법에 똑같이 쓴다. 따라서 특정 방법이 head 덕을 보지 않는다.

평가는 ADE20k~\cite{ade20k,ade20kijcv} 검증셋 $2{,}000$장 전체에서 하고, 프로브 학습에는 학습셋 $20{,}210$장 전체를 쓴다. $16{\times}16$ 격자라 절대 mIoU는 낮다. Table~\ref{tab:dense}가 이를 뒷받침한다. DINOv2-reg에서 레지스터 보호는 모든 축소율에서 mIoU를 높이며, 그 이득은 압축과 함께 커져 ${\sim}91\%$에서 $25.3$ 대 $19.6$으로 $+5.7$에 이른다. 분류 실험과 달리 여기서는 같은 개수의 무작위나 에너지, 고노름 기준이 무보호를 미미하게만 넘고 극단 압축에서는 무보호 수준 이하로 돌아가는 반면, 레지스터 보호만 모든 축소율에서 크게 앞선다. 밀집 설정에서 레지스터 이점이 더 뚜렷하다.

\begin{table}[t]
\centering
\caption{토큰 축소 하 ADE20k 선형 프로브 분할 mIoU(\%), DINOv2-ViT-B/14-reg4. 평가=검증셋
$2{,}000$장 전체, 프로브 학습=학습셋 $20{,}210$장 전체, $16{\times}16$ 격자라 절대 mIoU는 낮다. 전체 모델: $29.40$. 레지스터 보호만 모든 축소율에서 무보호 베이스라인을 넘는다.}
\label{tab:dense}
\begin{tabular}{lccccc}
\toprule
축소 & ToMe & \textbf{Ours} & 무작위 & 에너지 & 고노름 \\
\midrule
37\% & 28.50 & \textbf{29.00} & 28.50 & 28.52 & 28.57 \\
55\% & 27.06 & \textbf{28.85} & 27.42 & 27.46 & 27.55 \\
74\% & 24.96 & \textbf{27.83} & 25.15 & 25.13 & 25.34 \\
83\% & 22.96 & \textbf{27.10} & 23.37 & 23.37 & 23.45 \\
91\% & 19.59 & \textbf{25.29} & 18.98 & 19.29 & 19.54 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{보조 검증}
\label{sec:aux}
다음 검증들은 주 결과와 방향이 같다. 각 검증이 어떤 프로토콜을 쓰는지 함께 밝힌다.

\emph{seed와 토큰 수.} 병합과 kNN은 결정적이라 seed에 따른 분산이 매우 작다. 대신 평가셋을 부트스트랩으로 재표집해 불확실성을 잰다. 표준 train-갤러리에서 레지스터 이점의 $95\%$ 신뢰구간은 모든 축소율에서 $0$을 넘지 않으며, ${\sim}92\%$서 $[+7.0, +7.6]$, 중간 $55\%$서 $[+1.4, +1.8]$이다. 또 보호하는 레지스터 수를 $k{=}0$에서 $4$로 늘리면 정확도가 계단식으로 오르다 $k{\approx}3$에서 포화한다. ${\sim}92\%$서 $70.0$에서 $74.5$, $76.7$, $77.3$, $77.3$으로 오른다. 이득이 단순히 토큰을 더 남겨서가 아님을 보이며, 네 번째 레지스터를 더해도 이득이 없다. 같은 개수를 다른 기준으로 보호하는 것이 도움이 안 됨은 Table~\ref{tab:ablation}에서 이미 보였다.

\emph{다른 지표.} kNN이 특정 지표라는 우려에 답하려고 두 지표를 더 쟀다. 첫째, 같은 고정 특징에 선형 프로브~\cite{simclr}를 학습했다. 특징을 표준화한 뒤 클래스 평균에 최근접으로 분류하며 val을 층화 분할해 쓴다. 세 레지스터 모델 모두에서 레지스터 보호가 모든 축소율에서 앞서고, base 모델 기준 격차가 $37\%$의 $+1.2$에서 $91\%$의 $+6.3$으로 커진다. 둘째, 이미지 검색 평균정밀도 mAP~\cite{roxfordparis}도 쟀다. 격차가 $55\%$의 $+5.1$에서 ${\sim}91\%$의 $+16.4$로 커져 kNN 격차보다 크다. 검색 mAP는 검증셋 안에서 서로를 찾는 within-set 지표라 train 갤러리 개념이 없어 val에서 보고한다. kNN, 선형 프로브, 검색 mAP가 모두 같은 방향이라 이득이 특정 평가 방식이 아니라 특징 자체의 성질임을 시사한다.

\subsection{갤러리와 투표, 인코더에 대한 불변성}
\label{sec:consistency}
주 결과는 표준 train-갤러리 kNN을 쓴다.

결론이 갤러리 선택에 불변임을 확인하려고 val leave-one-out에서도 재측정했다. 작은 val 갤러리는 같은-클래스 이웃이 적어 절대 수치가 낮아 무압축이 $76.3$이지만, 방법 간 순서는 그대로이고 상대 격차는 오히려 더 크다. ${\sim}91\%$서 레지스터 보호가 ToMe 대비 $+10.2$, 공식 PiToMe 대비 $+8.2$로, train 갤러리의 $+7.3$, $+6.2$보다 크다. 이는 큰 train 갤러리가 더 관대하기 때문이다. 보호 기준 실험과 레지스터 개수 스윕도 val leave-one-out에서 같은 그림을 준다.

투표 방식에도 불변이다. train 갤러리를 공식 온도 가중 투표로 재채점하면 무압축 baseline이 $80.87$에서 $81.42$로 공인값 $82$에 더 다가서지만 순서는 그대로다. 인코더 크기에도 성립한다. DINOv2-ViT-S/14로 확장해도 순서는 보존되며 무압축 $77.4$, ${\sim}92\%$서 레지스터 보호가 ToMe 대비 $+4.4$이다. 다만 이득 크기는 인코더에 따라 달라 ViT-S가 세 레지스터 모델 중 가장 약하다.

\begin{table}[t]
\centering
\caption{val leave-one-out kNN에서의 재측정. gallery와 query 모두 val $50{,}000$장이고 각 이미지는 자기 자신을 뺀 이웃으로 분류하며 $k{=}20$, 무압축 $76.33$. train-갤러리 결과와 순서가 같고 상대 격차는 오히려 더 크다.}
\label{tab:consistency}
\begin{tabular}{lccc}
\toprule
토큰 축소 & ToMe & PiToMe & \textbf{Ours} \\
\midrule
37\% & 75.14 & 75.48 & \textbf{76.14} \\
55\% & 73.55 & 74.45 & \textbf{75.69} \\
74\% & 70.28 & 71.69 & \textbf{75.01} \\
83\% & 67.43 & 69.05 & \textbf{74.07} \\
91\% & 62.41 & 64.47 & \textbf{72.63} \\
\bottomrule
\end{tabular}
\end{table}


\section{적용 범위와 한계}
\label{sec:limitations}
\paragraph{레지스터 없는 인코더.} 레지스터 없는 DINOv2에서는 고노름 보호를 포함한 어떤 keep-prior도 베이스라인을 넘지 못하고 모두 ${\sim}0.2$%p 안에 머물며, 그 모델은 이미 병합에 강건해 $93\%$에서 ${\sim}4$%p만 잃는다. 따라서 우리는 주장을 레지스터를 가진 인코더로 제한하고, 고노름 유사물을 동작하는 방법으로 주장하지 \emph{않는다}.

\paragraph{모델 의존성.} 세 레지스터 모델의 보조 3-seed 비교에서 이득의 크기가 다르다. DINOv2-ViT-B/14에서는 극단 축소에서 레지스터 보호가 분명히 앞서 val-LOO $91\%$서 $+10.2$, train에서 $+7.3$이다. DINOv2-ViT-L/14에서는 무보호 베이스라인이 병합에 훨씬 취약해 $91\%$서 $2.8\%$로 붕괴하므로, 레지스터 보호는 절대값이 $8.6\%$로 낮아도 무보호 $2.8\%$ 대비 상대 이득이 $+5.8$로 크다. 반면 더 작은 DINOv2-ViT-S/14에서는 무보호가 이미 병합에 강건해 이득이 $91\%$서 $+0.9$로 작고, 동적 에너지 기준이 $61.5$ 대 $60.7$으로 레지스터 보호를 근소하게 앞선다. 따라서 효과는 모델 의존적이며 레지스터가 병목이 되는 극단 영역에서 가장 크다고 정직하게 제한한다.

\paragraph{규모와 불확실성.} 병합과 kNN은 결정적이나 특징추출은 GPU 비결정성으로 하위 실험 간 ${\sim}0.1$%p 변동이 있다. seed 반복 대신 평가셋 부트스트랩 CI로 유의성을 보이며, 이 CI는 중간 압축을 포함한 모든 축소율에서 $0$을 배제한다.


\section{결론}
\label{sec:conclusion}

우리는 레지스터 인지 토큰 축소를 제안했다. 이는 학습 없이 기존 인코더에 바로 얹는 플러그인으로, 토큰을
병합할 때 레지스터 토큰을 보호 집합에 넣어 병합에서 제외하고 패치만 합친다. 레지스터는 전역 정보를 흡수하도록
학습되므로, 이들을 지키면 일반적인 병합이 무심코 버리는 정보를 보존한다. 레지스터를 가진 DINOv2-ViT-B/14를
표준 train-갤러리 kNN으로 평가하면 이 방법은 모든 축소율에서 ToMe 계열 베이스라인을 앞서고, 그 격차는
압축이 심해질수록 벌어져 ${\sim}92\%$ 축소에서 top-1 정확도가 최대 $+7.3$%p 높다. 같은 개수를 보호하는
ablation은 이 이득이 토큰을 더 지켜서가 아니라 레지스터 자체에서 온다는 것을 분리해 보인다.

이 보호 규칙은 ToMe에만 국한되지 않는다. 같은 보호를 공식 PiToMe 병합 위에 얹어도 이득이 유지되고,
DINOv3와 ViT-5 같은 다른 레지스터 인코더로도 이어지며, 이런 모델에서는 레지스터를 제거하면 극한 축소에서
정확도가 붕괴해 레지스터가 병목임을 확인시킨다. 이득은 추가 계산 없이 얻어져 세 방법의 처리량이 사실상 같고,
선형 프로브와 검색 mAP, 부트스트랩 신뢰구간, 레지스터 개수 스윕, 그리고 val leave-one-out과 가중 투표,
ViT-S 재현까지 모두 같은 방향을 가리킨다. 다만 이득 크기는 인코더에 따라 달라 base와 large에서는 뚜렷하지만
small에서는 약하고, 레지스터가 병목이 되는 극단 압축에서 가장 크다. 레지스터 보호는 레지스터를 갖춘 어떤
인코더에도 값싸게 붙는 일반적 prior이며, 토큰 축소 연구가 레지스터를 일급 시민으로 다루도록 이끌기를 기대한다.

% ---------------------------------------------------------------
\bibliographystyle{splncs04}
\bibliography{main}

\end{document}
