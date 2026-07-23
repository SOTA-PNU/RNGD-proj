\documentclass[runningheads]{llncs}

% ###################################################################
% ⚠️ OLD SNAPSHOT (2026-07-01). 이 파일은 옛 영어 초안입니다.
%    ⇒ 현재 동기화된 영어본은 main_en.md 입니다(2026-07-02, 한국어 main.md와 대등).
%    이 백업은 이력 보존용. 이력·체크리스트: ACCV/영어본_동기화_상태.md 참조.
% ###################################################################

% ---------------------------------------------------------------
% Include basic ACCV package
 
% TODO REVIEW: Insert your submission number below by replacing '*****'
% TODO FINAL: Comment out the following line for the camera-ready version
\usepackage[review,year=2026,ID=*****]{accv}
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

% ---------------------------------------------------------------
% TODO: 간이 초안 제목 (확정 전)
\title{Don't Merge the Registers: Register-Aware Token Reduction for
Vision Encoders with Registers at Extreme Compression}

\titlerunning{Register-Aware Token Reduction}

% 익명 제출(review). TODO FINAL: 실제 저자/소속.
\author{Anonymous ACCV submission}
\authorrunning{Anonymous ACCV submission}
\institute{Paper ID *****}

\maketitle


\begin{abstract}
% 간이 초안 (약 150단어 목표). 실험 수치는 예비(preliminary).
Foundation vision encoders such as DINOv2 and CLIP are widely reused as
off-the-shelf feature extractors, but processing hundreds of tokens makes
inference costly. Token reduction (merging or pruning) lowers this cost, yet
existing methods rank tokens by generic similarity or energy heuristics. Some
of these encoders concentrate global information in a few \emph{register}
tokens~\cite{registers}, and we observe that a standard size-weighted merging
baseline (ToMe-style) merges these tokens away under \emph{extreme} reduction,
with a large accompanying accuracy drop. We study a simple, training-free
change: protect the register tokens and aggressively merge the remaining
patch tokens. On a single register-equipped encoder (DINOv2-ViT-B/14 with $4$
registers), evaluated by kNN top-1 on ImageNet-1k (single seed), protecting
registers retains $71.98\%$ at ${\sim}91\%$ token reduction versus $63.9\%$ for
the same merging without protection ($+8.1\%$), and an equal-count keep-prior
ablation (random, energy-based, or high-norm) stays within measurement noise of
the unprotected baseline. Unlike RegCache (which prefixes registers for
\emph{quantization}) and FNA (which \emph{approximates} attention)---both of
which keep all tokens---we use registers as a \emph{keep-prior} for
sequence-length \emph{reduction}. On ADE20k linear-probe segmentation the same
protection improves mIoU at every reduction ratio (e.g.\ $+3.1$ at ${\sim}91\%$),
where the alternative keep-priors do not help. We report this as a
single-encoder (DINOv2-ViT-B/14), single-seed study; additional encoders,
retrieval, and on-hardware speedup are future work.
  \keywords{Token reduction \and Vision transformers \and Register tokens
\and Efficient inference}
\end{abstract}


\section{Introduction}
\label{sec:intro}

% 간이 초안. [TODO]는 GPU 본실험 후 채움.
Self-supervised foundation encoders (DINOv2, CLIP, SigLIP) are now used as
frozen, off-the-shelf backbones for classification, retrieval, and dense
prediction. Their cost grows with the number of tokens, so \emph{token
reduction}---merging or pruning tokens inside the network---is an attractive,
training-free way to speed them up.

Existing token-reduction methods decide \emph{which} tokens to keep using
generic signals: feature similarity (ToMe~\cite{tome}) or spectral/cluster
energy (PiToMe~\cite{pitome}). However, recent work shows that vision
transformers develop a small number of \emph{register} tokens (when trained
with them~\cite{registers}) or \emph{high-norm artifact} tokens (when not),
which act as global-information stores and behave very differently from
ordinary patch tokens~\cite{massive}. For a register-equipped encoder we
observe that a standard merging baseline removes these register tokens under
\emph{extreme} reduction ($>90\%$), accompanied by a large accuracy drop.

We study \textbf{register-aware token reduction}, a training-free plug-in
that (i) protects the explicit register tokens (for register-free encoders a
high-norm analogue is a natural but here-untested extension), (ii) excludes
them from merging, and (iii) aggressively merges the remaining redundant
patches. On the encoder we test, this lets the model tolerate much higher
reduction ratios. Our contributions are:
\begin{itemize}
  \item On a register-equipped encoder we observe that a standard merging
  baseline removes register tokens under extreme reduction, with a large
  accuracy drop; protecting them recovers most of it.
  \item We evaluate a simple, training-free register-aware protection rule and
  find it robust to extreme reduction on the tested encoder.
  \item On DINOv2-ViT-B/14 ($4$ registers), ImageNet-1k kNN top-1, single seed,
  protecting registers improves over the unprotected baseline by $+8.1\%$ at
  ${\sim}91\%$ reduction, with the gain increasing as reduction increases
  ($+1.0\!\to\!+8.1$). [TODO: multi-seed, linear-probe, more models, retrieval,
  dense prediction, latency.]
  \item An equal-count keep-prior ablation shows random, energy-based, and
  high-norm protection stay within measurement noise of the unprotected
  baseline, whereas register protection does not; this points to registers
  (not merely protecting more tokens) as the source of the gain. The energy
  baseline is a keep-prior proxy, not the PiToMe method.
\end{itemize}


\section{Related Work}
\label{sec:related}

% 간이.
\paragraph{Token reduction.} ToMe~\cite{tome} merges similar tokens via
bipartite soft matching with size-weighted attention; DynamicViT, EViT and
PiToMe~\cite{pitome} refine \emph{which} tokens to keep using learned masks or
spectral energy. These methods are model-agnostic and do not use
register/artifact-token structure. [TODO: expand with dense-prediction merging
(ALGM, ClustViT).]

\paragraph{Register and artifact tokens.} Darcet~\etal~\cite{registers} show
that ViTs benefit from explicit register tokens that absorb high-norm artifact
tokens; \cite{massive} characterize these as ``massive activations.'' These
works study representation quality, not efficiency.

\paragraph{Exploiting artifact tokens for efficiency.} Closest to us, RegCache
prevents outliers by prefixing registers for activation \emph{quantization},
and FNA~\cite{fna} uses massive/artifact tokens as landmarks to
\emph{approximate} attention in linear time. Both keep all tokens; we instead
\emph{reduce sequence length}, using registers as a keep-prior. [TODO: precise
positioning vs RegCache/FNA.]


\section{Method}
\label{sec:method}

% 간이. 표기는 단순화.
Let a ViT process tokens $X=[x_1,\dots,x_T]$ with a prefix of $p$ global tokens
(class token plus $p-1$ registers). Standard token merging at each block
selects $r$ token pairs by similarity and merges them with size-weighted
averaging, protecting only the class token.

\textbf{Register-aware protection.} We enlarge the protected set from the class
token alone ($p=1$) to the class token \emph{plus all register tokens}
($p=1+\#\text{registers}$). For encoders without explicit registers the natural
analogue protects the top-$k$ high-norm tokens; we find such encoders are
already robust to merging, so this is a scope limitation rather than a gain
(Sec.~\ref{sec:exp}). Protected tokens are excluded from merging (kept aside and passed through);
only the remaining patch tokens are merged by the same size-weighted bipartite
soft matching (a ToMe-style baseline; we do not apply ToMe's
proportional-attention bias to either arm). The method is training-free.
For dense prediction, merged tokens can be un-merged back to full resolution.
The main comparison omits ToMe's proportional-attention bias on \emph{both} arms
(a controlled, like-for-like setting); we separately verify against a faithful
ToMe that includes it (Sec.~\ref{sec:exp}).
[TODO: formal notation, unmerge for dense tasks.]


\section{Experiments}
\label{sec:exp}

% 간이 — 현재 수치는 예비(preliminary, CPU). GPU 본실험은 진행 중.
\paragraph{Setup.} We evaluate a frozen DINOv2-ViT-B/14 (with $4$ registers) on
ImageNet-1k by $k$-NN classification ($k{=}20$, no training), using the
$50{,}000$-image validation set as both gallery and query (leave-one-out) at
$224{\times}224$ input. This operating point is lower-resolution than the
standard $518$ DINOv2 protocol, so absolute numbers should be read accordingly.
Token reduction merges $r$ tokens per block with a size-weighted, ToMe-style
baseline; ``ToMe'' protects the class token only, ``Ours'' additionally protects
the register tokens. All results are single-seed. [TODO: multi-seed,
linear-probe / standard-gallery kNN, retrieval, segmentation/depth, latency.]

\paragraph{Main result (DINOv2-with-registers, full ImageNet-1k val).}
Table~\ref{tab:main} and Figure~\ref{fig:result} (left) report $k$-NN accuracy
under size-weighted merging on all $50{,}000$ validation images. Protecting
registers improves accuracy at every measured reduction ratio, and the gain
increases with compression, from $+1.0$ at $37\%$ reduction to $+8.1$ at
${\sim}91\%$. At the most aggressive setting our method retains $71.98\%$ ($4.4$
points below the full model) while the unprotected baseline drops to $63.91\%$
($12.4$ points below). This ${\sim}8$-point gain is two orders of magnitude
above the run-to-run noise floor (below); the per-ratio fine structure is
single-seed and we do not test it for significance.

\begin{table}[t]
\centering
\caption{$k$-NN top-1 accuracy (\%) on the full ImageNet-1k validation set
($50{,}000$ images, $k{=}20$), DINOv2-ViT-B/14 with $4$ registers, size-weighted
ToMe. Full model: $76.35$. ``ToMe'' protects the class token only; ``Ours'' also
protects the register tokens. Reduction \% is the nominal per-block schedule;
the realized reduction is ${\sim}1$ point lower, and at the extreme setting Ours
retains ${\sim}2$ more tokens (the protected registers) than ToMe (single seed).}
\label{tab:main}
\begin{tabular}{lccc}
\toprule
Token reduction & ToMe (protect CLS) & Ours (protect CLS+reg) & $\Delta$ \\
\midrule
37\% & 74.70 & \textbf{75.71} & $+1.01$ \\
55\% & 72.93 & \textbf{75.30} & $+2.38$ \\
74\% & 70.34 & \textbf{74.13} & $+3.79$ \\
83\% & 67.62 & \textbf{73.28} & $+5.66$ \\
92\% & 63.91 & \textbf{71.98} & $+8.07$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_result.png}
\caption{\textbf{Left:} $k$-NN accuracy vs.\ token reduction on DINOv2-reg
(single seed); the unprotected ToMe-style baseline drops sharply under extreme
reduction while register-aware protection stays close to the full model.
\textbf{Right:} the gap $\Delta=$~Ours$-$ToMe grows with compression for
DINOv2-reg; for a register-free DINOv2 Ours${=}$ToMe by construction
($\Delta{\approx}0$), a noise-floor check. Causal attribution to registers
comes from the keep-prior ablation (Fig.~\ref{fig:ablation}), not this panel.}
\label{fig:result}
\end{figure}

\paragraph{Noise-floor / sanity control (register-free encoder).}
Table~\ref{tab:control} repeats the experiment on a DINOv2 \emph{without}
registers (prefix${=}1$), where ``Ours'' and ``ToMe'' are the \emph{same}
computation; $\Delta$ is $0$ by construction, and the small non-zero values
($\le0.05\%$) come from re-run non-determinism. This establishes a
measurement-noise floor and rules out a harness artifact, but it does
\emph{not} by itself attribute the gain to registers---that attribution rests
on the equal-count keep-prior ablation below. Note also that this register-free
model is intrinsically more robust to merging (its unprotected baseline loses
only ${\sim}4$ points at $93\%$ vs.\ $12.4$ for the register model), so the two
models differ in more than the registers.

\begin{table}[t]
\centering
\caption{Negative control: DINOv2-ViT-B/14 \emph{without} registers
(prefix${=}1$), full ImageNet-1k val. With no registers to protect, Ours${=}$ToMe
and the gain vanishes. Full model: $75.85$.}
\label{tab:control}
\begin{tabular}{lccc}
\toprule
Token reduction & ToMe & Ours & $\Delta$ \\
\midrule
37\% & 75.17 & 75.19 & $+0.02$ \\
56\% & 74.55 & 74.57 & $+0.01$ \\
75\% & 73.77 & 73.73 & $-0.05$ \\
84\% & 73.01 & 73.04 & $+0.03$ \\
93\% & 71.69 & 71.68 & $-0.01$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Only registers help: keep-prior ablation.}
A natural objection is that protecting any extra tokens, or any
importance-based keep-prior, would help equally. Table~\ref{tab:ablation}
argues against this: holding the merging fixed and varying only \emph{which}
equal-sized set is protected, only the registers help. At ${\sim}91\%$
reduction, protecting the same number of tokens chosen at \emph{random}
($63.2\%$), by a mean-similarity \emph{energy} keep-prior ($64.0\%$), or by
highest \emph{norm} ($63.8\%$) all stay within ${\sim}0.9$ point of the
unprotected baseline ($63.9\%$), whereas protecting registers gives $71.98\%$
($+7.9$ over the best alternative). These non-register priors fall within the
run-to-run noise floor of the baseline, so we do not claim an ordering among
them. An important caveat qualifies this. The random/energy/high-norm sets in
Table~\ref{tab:ablation} are selected once from the input-layer embedding and
held fixed, whereas high-norm artifact tokens emerge mid-network. In a check at $n{=}10{,}000$
(base model, $3$ seeds, kNN), \emph{dynamically} re-selecting the high-norm or
energy set at every block makes those baselines stronger: at ${\sim}91\%$
reduction the best dynamic prior reaches $53.3\%$, versus ${\sim}49\%$ for the
static priors and the unprotected baseline and $57.8\%$ for register
protection---i.e.\ it closes about half the gap, but register protection still
wins by ${\sim}+4.5$ (single seed). Register protection remains the strongest
keep-prior at \emph{extreme} reduction, but the reading ``only registers help''
holds for the \emph{static} priors of Table~\ref{tab:ablation}, not for dynamic
ones; a full-scale comparison against dynamically re-selected priors is still
needed. We \emph{did} compare against the \emph{actual} PiToMe algorithm (its
official energy-based merging, re-scored per layer---not our proxy): at $n{=}3000$
on DINOv2-reg, register protection beats PiToMe at every ratio
($+1.1$/$+0.7$/$+4.1$ at $55$/$74$/$91\%$ reduction), while PiToMe itself clearly
beats the unprotected baseline ($56.0$ vs.\ $52.6$ at $91\%$); larger-scale
multi-seed confirmation is pending. To rule out that our baseline is a weak
strawman, we also re-implemented a \emph{faithful} ToMe---with the
proportional-attention bias, attention-key similarity, and merging between the
attention and MLP sub-blocks---as the unprotected arm: register protection still
wins at every ratio ($+2.1$/$+3.0$/$+10.0$ at $55$/$74$/$91\%$ reduction;
$n{=}3000$, single seed), confirming the advantage is not an artifact of a
weakened baseline. Two further checks address the single-seed and token-count
concerns without GPU access. First, because the merging and $k$-NN are
deterministic on CPU (zero seed variance), we instead bootstrap-resample the
evaluation set ($n{=}3000$): the register advantage at ${\sim}91\%$ reduction has
a $95\%$ confidence interval of $[+5.7, +9.2]$ (excluding zero), whereas at
moderate ($55\%$) reduction the interval includes zero---consistent with our
claim being specific to extreme compression. Second, sweeping the number of
protected registers ($k{=}0{\to}4$, same merging) shows the gain is not a mere
token-count artifact: protecting a single register barely helps, but protecting
two or more produces a step increase that then saturates, indicating the
registers must be preserved as a group rather than each contributing
independently. The advantage is also not specific to the $k$-NN metric: under
image-retrieval mean average precision (mAP) on the same features
($n{=}3000$), register protection wins at every ratio, with the gap \emph{growing}
from $+3.1$ at $37\%$ to $+14.2$ at ${\sim}91\%$ reduction---larger than the
$k$-NN gap---indicating the effect is a property of the features rather than of a
particular evaluation. Finally, the effect is model-dependent: in a
two-model check register protection wins clearly on DINOv2-ViT-B/14 at extreme
reduction, but on the smaller DINOv2-ViT-S/14 a dynamic energy prior matches it,
so we do not overstate universality.

\begin{table}[t]
\centering
\caption{Keep-prior ablation on DINOv2-reg, full ImageNet-1k val
($50{,}000$, $k{=}20$). Same size-weighted merging; only the protected set
changes, each protecting the same number of tokens beyond the class token.
Only register protection improves over the unprotected baseline; random, energy
(a mean-similarity proxy, not PiToMe), and high-norm keep-priors stay within its
noise floor. Keep-prior sets are chosen once at the input layer; single seed.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Reduction & ToMe & \textbf{Ours (reg)} & Random & Energy & High-norm \\
\midrule
37\% & 74.70 & \textbf{75.71} & 74.70 & 74.58 & 74.78 \\
55\% & 72.95 & \textbf{75.30} & 72.82 & 72.70 & 72.87 \\
74\% & 70.36 & \textbf{74.14} & 70.00 & 70.16 & 70.16 \\
83\% & 67.60 & \textbf{73.25} & 67.26 & 67.51 & 67.53 \\
91\% & 63.90 & \textbf{71.98} & 63.19 & 64.04 & 63.84 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.74\linewidth]{fig_ablation.png}
\caption{Keep-prior ablation on DINOv2-reg (single seed): among equal-count
protected sets, only the registers improve over the unprotected baseline;
random, energy (a mean-similarity proxy, not PiToMe), and high-norm protection
all stay within its noise floor, and the register advantage widens with
compression.}
\label{fig:ablation}
\end{figure}

\paragraph{Scope (register-free encoders).} On a register-free DINOv2, no
keep-prior---including high-norm protection---improves over the baseline (all
within ${\sim}0.2$ point), and that model is already robust to merging (its
baseline loses only ${\sim}4$ points at $93\%$). We therefore restrict our claim
to register-equipped encoders and do \emph{not} claim the high-norm analogue as
a working method; why register-free encoders show neither the fragility nor a
benefit from high-norm protection is left open.

\paragraph{Dense prediction (ADE20k segmentation).} Because dense tasks depend
on per-token spatial information, register protection should matter more there.
We train a linear segmentation probe on frozen full features and evaluate mIoU
under token reduction, un-merging tokens back to the patch grid
($2{,}000$ train / $2{,}000$ val, a $16{\times}16$ grid, so absolute mIoU is
low). Table~\ref{tab:dense} confirms this: on DINOv2-reg, register protection
improves mIoU at \emph{every} reduction ratio and the gap widens with
compression, reaching $+3.1$ mIoU at ${\sim}91\%$ ($17.2$ vs.\ $14.1$). Notably,
unlike the classification ablation, here the equal-count random/energy/high-norm
keep-priors do \emph{not} help---they match or fall below the unprotected
baseline---so on dense prediction the register advantage is cleaner. (On a
register-free DINOv2 all keep-priors coincide.)

\begin{table}[t]
\centering
\caption{ADE20k linear-probe segmentation mIoU (\%) under token reduction,
DINOv2-ViT-B/14-reg4 ($2{,}000$/$2{,}000$ train/val, $16{\times}16$ grid; low
absolute mIoU is the low-resolution operating point). Full model: $23.10$. Only
register protection improves over the unprotected baseline.}
\label{tab:dense}
\begin{tabular}{lccccc}
\toprule
Reduction & ToMe & \textbf{Ours (reg)} & Random & Energy & High-norm \\
\midrule
37\% & 21.12 & \textbf{22.42} & 20.92 & 20.70 & 20.86 \\
55\% & 19.31 & \textbf{21.71} & 19.14 & 19.10 & 19.02 \\
74\% & 17.32 & \textbf{20.43} & 16.86 & 16.85 & 16.51 \\
83\% & 15.71 & \textbf{19.12} & 15.27 & 15.31 & 15.48 \\
91\% & 14.05 & \textbf{17.19} & 13.44 & 13.10 & 13.32 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Mechanism: the baseline merges registers early.} We directly track
merging under the unprotected baseline. On DINOv2-reg at $r{=}16$ ($74\%$
reduction), across $16$ images $94\%$ of register tokens are merged into other
tokens, at a mean first-merge block of ${\sim}3.6$ of $12$ (as early as block
$0$); only $6\%$ survive to the final layer. Protecting them (Ours) keeps all
registers by construction. This supports the interpretation that the gain comes
from preventing early destruction of the register tokens rather than from an
indirect effect, though it is measured on a small image sample and single $r$.

\paragraph{Efficiency: FLOPs, not measured latency.} Token reduction lowers the
backbone FLOPs by $17$--$43\%$ over our range (e.g.\ $43\%$ at ${\sim}91\%$
reduction; the reduction is sub-linear because tokens are removed gradually and
projection/MLP FLOPs dominate). Ours and the unprotected baseline have
essentially identical FLOPs ($<0.1\%$ difference, i.e.\ the ${\sim}2$ extra
retained registers are negligible), so the accuracy gain is \emph{not} bought
with extra compute. We caution that this FLOP reduction did \emph{not} translate
into a wall-clock speedup on the one accelerator we measured: on a commercial
NPU, per-forward latency was flat across reduction ratios (overhead-bound at
these token counts), so we make no on-chip acceleration claim; GPU throughput is
future work.

\paragraph{Pending.} Full-scale ($50$k) multi-seed error bars, linear-probe, and
additional register-equipped encoders; a full-scale dynamically re-selected
keep-prior ablation; and image--text retrieval are in progress.


\section{Conclusion}
\label{sec:conclusion}

% 간이.
We studied register-aware token reduction, a training-free scheme that protects
register tokens during merging. On a single register-equipped encoder
(DINOv2-ViT-B/14), evaluated by kNN, it improves over a ToMe-style baseline by
up to $+8.1$ points at ${\sim}91\%$ token reduction, and an equal-count
keep-prior ablation attributes the gain to the registers rather than to merely
protecting more tokens. We report this as a single-encoder, single-seed,
single-metric finding; multi-seed statistics, a standard linear-probe/gallery
protocol, additional register-equipped encoders, a head-to-head with PiToMe,
retrieval, dense prediction, and on-hardware latency are needed to establish
generality and efficiency. [TODO.]


% ---------------------------------------------------------------
\bibliographystyle{splncs04}
\bibliography{main}

\end{document}
