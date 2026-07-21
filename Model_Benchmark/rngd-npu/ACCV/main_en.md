\documentclass[runningheads]{llncs}

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

\maketitle


\begin{abstract}
Foundation vision encoders such as DINOv2 and CLIP are widely used as frozen, off-the-shelf feature extractors, but the hundreds of tokens they process per image make inference costly. Token reduction lowers this cost by merging or pruning tokens inside the network without retraining, yet existing methods decide which tokens to keep from generic, model-agnostic signals such as feature similarity or spectral energy, remaining blind to token identity. Some of these encoders, however, concentrate global information in a few register tokens whose unusually large norms turn them into attention sinks. We show that standard size-weighted merging tends to destroy exactly these tokens: under extreme compression it absorbs the registers into ordinary patches early, and accuracy collapses---on DINOv2-ViT-B/14 from $80.9\%$ uncompressed, close to the published $82\%$, to $70.0\%$ at ${\sim}92\%$ token reduction. Motivated by this, we propose register-aware token reduction, a simple training-free rule that adds the registers to the merger's protected set and aggressively merges only the remaining patches. It introduces no new merging algorithm and no extra computation. Under matched budgets it raises top-1 accuracy by $+7.3$ over ToMe and $+6.2$ over PiToMe at ${\sim}92\%$ reduction, and the gain grows with compression; an equal-count ablation attributes it to the registers themselves rather than to keeping more tokens. The same protection also helps on top of PiToMe's merger, transfers to the rotary-position-embedding encoders DINOv3 and ViT-5---where removing the registers collapses accuracy---and to ADE20k linear-probe segmentation. Since the method is deterministic, we quantify uncertainty by bootstrapping the evaluation set, and the $95\%$ confidence interval of the gain excludes zero at every reduction ratio.
  \keywords{Token reduction \and Vision transformers \and Register tokens \and Efficient inference}
\end{abstract}

\section{Introduction}
\label{sec:intro}

% P1. Problem context
Foundation vision encoders such as DINOv2~\cite{dinov2} and CLIP~\cite{clip}
have become the default backbones for classification, retrieval, and dense
prediction, typically deployed as frozen feature extractors. These Vision
Transformers (ViTs)~\cite{vit} process an image as hundreds of
tokens---patches cut from a regular grid, plus a few special tokens such as
the class token that summarizes the whole image---and since the cost of
self-attention grows quadratically with the number of tokens, inference slows
down rapidly as resolution and model size increase. Token
reduction~\cite{tome,pitome,evit,dynamicvit}, which merges or prunes similar
or unimportant tokens inside the network without retraining, is the standard
remedy.

% P2. Gap: homogeneous rules, non-homogeneous tokens
The central decision in token reduction is which tokens to keep and which to
merge, and existing methods make this decision with model-agnostic, generic
signals. ToMe~\cite{tome} selects the most similar token pairs by the cosine
similarity of their attention keys and merges them by averaging;
PiToMe~\cite{pitome} scores token importance with the spectral energy of a
token-similarity graph; DynamicViT and EViT~\cite{dynamicvit,evit} use learned
importance. Recent work, however, shows that ViT tokens are not homogeneous:
encoders trained with registers concentrate global information in a handful
of register tokens~\cite{registers}, whose exceptionally large norms make
them attention sinks~\cite{massive}. The architecture itself thus signals
which tokens are special, yet existing reduction rules look only at
similarity or energy scores and remain blind to token identity.

% P3. Observation: failure under extreme compression
We observe that this mismatch becomes a concrete failure under extreme
compression. Standard merging, which sees only similarity and size,
prematurely merges these few global tokens into ordinary patches as
compression grows. Tracking an unprotected merge schedule shows that most
registers are already absorbed into patch averages in the first half of the
network; as a result, on the standard DINOv2 k-NN
evaluation protocol~\cite{dino} (ImageNet-1k, full training set as the
reference dataset), accuracy collapses from $80.9\%$ without compression to
$70.0\%$ at ${\sim}92\%$ token reduction with DINOv2-ViT-B/14.


% P4. Proposal and positioning
Motivated by this observation, we study register-aware token reduction. The
rule is simple: widen the protected set from the class token alone to include
the registers, exempt them from merging, and merge the remaining patch tokens
aggressively. It requires no training and leaves the merging algorithm
unchanged. It is not a new merging method but a protection-rule that decides
\emph{what to protect} within the merging framework, parallel to how PiToMe
adds an energy rule in the same slot.
It is also distinct from prior work that exploits register-like tokens for efficiency: RegCache~\cite{regcache} injects auxiliary prefix registers to absorb activation outliers for quantization, and FNA~\cite{fna} uses massive and artifact tokens to approximate self-attention in linear time. Neither uses these tokens to shorten the sequence; we instead use registers as a protection criterion for reducing sequence length.


% P5. Evidence summary
We validate this rule from several directions. In controlled comparisons
under the same harness and the same budget, register protection gains $+7.3$
over ToMe and $+6.2$ over PiToMe at ${\sim}92\%$ reduction, and the gains
grow with compression. Ablations that protect the same number of tokens
chosen at random, by energy, or by high norm stay near the unprotected
baseline, showing that the benefit comes from the registers themselves rather
than from keeping more tokens. Adding the same protection on top of the
PiToMe merger preserves the gains, and the rule carries over to
rotary-position-embedding (RoPE) register encoders---DINOv3~\cite{dinov3} and
ViT-5~\cite{vit5}---as well as to ADE20k linear-probe segmentation. Since the
method is deterministic, we quantify uncertainty by bootstrapping the
evaluation set; the $95\%$ confidence intervals of the gains exclude $0$ at
every reduction ratio.

% P6. Contributions
In summary, our contributions are:
\begin{itemize}
  \item We identify and quantify a failure mode of standard size-weighted
        token merging: under extreme compression it destroys register tokens
        early, and this destruction drives the accuracy collapse.
  \item We propose a simple, training-free protection-rule that places registers in
        the merger's protected set. It is agnostic to the merging algorithm
        and adds no computation.
  \item We validate the rule across two mergers (ToMe, PiToMe), three encoder
        families (DINOv2, DINOv3, ViT-5), and two tasks (classification,
        segmentation), and support the mechanism with merge-tracking and
        attention analyses.
\end{itemize}


\section{Related Work}
\label{sec:related}

\paragraph{Token reduction.}
ToMe~\cite{tome} merges similar tokens without training via size-weighted
bipartite soft matching on attention-key similarity, and established the
framework that subsequent token-reduction work builds on. Within this
framework, the remaining question is which tokens to merge and which to
keep: PiToMe~\cite{pitome} retains the bipartite matching and adds a
spectral-energy criterion, while DynamicViT and EViT~\cite{dynamicvit,evit}
use learned masks. Many other variants have been explored and are compared systematically by Haurum \etal~\cite{haurum}. In all of them, however, the decision rule relies on
model-agnostic, generic signals---similarity, energy, or learned
importance---and none exploits the register or outlier-token structure of
the encoder. We propose a new rule within the same merging framework that
takes the registers, signaled by the model architecture itself, as the
protection criterion---occupying the same slot in which PiToMe adds its
energy rule.

\paragraph{Registers and outlier tokens.}
Darcet \etal~\cite{registers} found that ViTs repurpose a few low-information
background patches as high-norm outlier tokens for internal computation, and
that adding dedicated register tokens absorbs this behavior, smoothing
feature and attention maps and improving dense prediction. Sun
\etal~\cite{massive} characterize the same phenomenon as ``massive
activations.'' These outlier tokens also degrade dense prediction~\cite{denoisingvit}, and register behavior can be
reproduced by injecting tokens at test time without
training~\cite{testtimeregisters}. Throughout this literature, registers are
a device for representation quality or a subject of analysis; none of it asks
what should happen to these tokens when the sequence itself is
\emph{compressed}. We take two facts from it---global information
concentrates in a few identifiable tokens, and destroying such components is
dangerous---and reinterpret registers as a \emph{protection criterion} for
token reduction.

\paragraph{Outlier tokens for efficiency.}
Closest to ours are two works that exploit the same phenomenon for
efficiency, along different axes. RegCache~\cite{regcache} injects auxiliary,
semantically empty prefix registers so that activation outliers land on them,
enabling post-training quantization; the injected tokens are later deleted.
FNA~\cite{fna} uses the structured attention patterns formed by massive and
artifact tokens as landmarks to approximate self-attention in linear time.
Neither uses these tokens to shorten the sequence: RegCache reduces bit
width and FNA reduces per-token attention cost, while the token count stays
intact. We target the third axis---sequence length---and use registers as the
protection criterion that makes extreme reduction viable. More broadly,
sequence-length reduction is complementary and orthogonal to the line of work
that accelerates the attention operation itself, whether
exactly~\cite{flashattention} or by
approximation~\cite{linformer,performer,reformer,nystromformer}.

% \section{Related Work}
% \label{sec:related}

% \paragraph{Token reduction.} ToMe merges similar tokens without training via
% size-weighted bipartite soft matching, establishing the framework that later token-reduction
% work builds on. The open question in this framework is \emph{which} tokens to merge or keep: PiToMe keeps
% ToMe's matching and adds a spectral-energy rule, while
% DynamicViT and EViT use learned masks. Many other variants have been
% studied, including adaptive token sampling~\cite{ats}, adaptive halting~\cite{avit},
% slow-fast token evolution~\cite{evovit}, learned compression rates~\cite{diffrate},
% unified pruning-and-merging~\cite{tofu,ltmp}, learned merging/pooling~\cite{tokenmerger,tokenpooling},
% and a small set of learned tokens~\cite{tokenlearner}, and a systematic
% comparison is given by Haurum et al.~\cite{haurum}. All these rules use
% model-agnostic signals (similarity, energy) and do not exploit register/outlier-token structure. Within the same merging framework, we propose a rule that retains the registers indicated by the model's own structure, the same place where PiToMe adds an energy rule.

% \paragraph{Registers and outlier tokens.} Darcet et al. found
% that ViTs place high-norm outlier tokens at low-information background positions
% and repurpose them for internal computation, and added dedicated register tokens
% to absorb them, smoothing feature and attention maps and improving dense
% prediction. Sun et al. characterize these as ``massive activations.'' The
% goal of these works is representation quality, not efficiency; we reinterpret the
% registers as a \emph{keep-prior for compression}. The same high-norm phenomenon also arises
% in large language models, concentrated in a few channels or tokens where it
% complicates quantization~\cite{llmint8,smoothquant} and produces attention sinks
% that concentrate attention on specific tokens~\cite{streamingllm,attnsinkemerge,quantizabletransformers};
% removing these components collapses performance~\cite{bertbusters,outlierfreq}.
% In vision, these artifacts likewise degrade dense prediction~\cite{denoisingvit},
% and register-like behavior can be reproduced without training by injecting a
% register token at test time~\cite{testtimeregisters}.

% \paragraph{Exploiting outlier tokens for efficiency.} Closest to us,
% RegCache prefixes registers to suppress outliers for activation
% \emph{quantization}, and FNA uses massive/outlier tokens as landmarks
% to \emph{approximate} attention in linear time. Both keep all tokens. We instead
% use registers as a keep-prior to \emph{reduce the sequence length}. More broadly, our
% sequence-length reduction is complementary and orthogonal to methods that
% approximate or linearize the attention operation
% itself~\cite{flashattention,linformer,performer,reformer,nystromformer}.


\section{Register-Aware Token Reduction}
\label{sec:method}

\paragraph{Preliminaries: Merging with a Protected Set.}
A ViT processes a token sequence $X=[x_1,\dots,x_T]$ whose first $p$ tokens
are global: the class token and $p{-}1$ registers. Token merging as
introduced by ToMe~\cite{tome} inserts a merging step between the attention
and the MLP of every block: tokens are split into two alternating sets, each
token in one set is matched to its most similar counterpart in the other by
the cosine similarity of their attention keys, and the $r$ highest-scoring
pairs are merged by a size-weighted average. A merged token carries its
\emph{size}---the number of original tokens it represents---and attention
logits are rescaled by size (\emph{proportional attention}). The framework
already contains a protected set $\mathcal{P}$ of tokens exempt from
matching; in ToMe, $\mathcal{P}=\{\text{CLS}\}$. PiToMe~\cite{pitome}
retains this machinery and replaces the pair-selection criterion with a
spectral-energy score; the protected set is left untouched.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_method.pdf}
\caption{Register-aware token reduction. Each block applies attention,
merging, and MLP; patch tokens are merged $r$ pairs at a time by
size-weighted bipartite soft matching, while the protected set is exempt
from merging and passes through unchanged. Top: ToMe protects only the class
token, so under extreme compression the registers are prematurely merged
away. Bottom: ours widens the protected set to include the registers,
preserving them to the last block.}
\label{fig:method}
\end{figure}

\paragraph{The Protection Rule.}
We widen the protected set to the full prefix,
\begin{equation}
\mathcal{P}=\{\text{CLS}\}\cup\{\text{registers}\},
\end{equation}
so that registers pass through every merging step unchanged while the
remaining patch tokens are merged as aggressively as before
(Fig.~\ref{fig:method}). The rule is training-free and adds no computation.
It modifies neither the matching nor the averaging, so it is agnostic to the
merger and applies unchanged to ToMe and PiToMe alike.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_attention.pdf}
\caption{How strongly the class token attends to each patch, shown by color: red is high and blue is low, with the four panels on a common scale. In a trained DINOv2-reg a few high-norm registers act as attention sinks that absorb attention. Under extreme compression at ${\sim}92\%$, ToMe and PiToMe keep only the class token and merge the registers away, so this concentration pattern is disturbed, while Ours protects the registers and keeps a pattern close to the uncompressed one. A qualitative example on a single image at the last block.}
\label{fig:attention}
\end{figure}

\paragraph{Why Registers Must Be Protected.}
Registers are trained, image-independent tokens that condense global rather
than local information: linear probes on them recover class labels well but
spatial positions poorly~\cite{registers}, and in large ViTs the global
representation is in fact dominated by the register
tokens~\cite{clsregdecouple}, whose exceptionally large norms make them
attention sinks that absorb a substantial share of the class token's
attention (Fig.~\ref{fig:attention}). Merging, by contrast, is
identity-blind: the matching sees only key similarity and size, and treats a
register like any other token. Under moderate compression this rarely
matters, because redundant patches suffice to fill the merging budget. Under
extreme compression, however, registers become likely match candidates, and
once a register is averaged into a group of patches its content is diluted
and cannot be recovered; the global information it carried is lost even when
the final representation is read from the class token alone. We verify this
account directly: tracking an unprotected schedule shows
that most registers are absorbed in the first half of the network, and
protecting the same number of tokens chosen by any other criterion fails to
recover the accuracy.

\paragraph{Implementation.}
We realize the rule inside the same faithful harness as the original methods, changing
only the protected set, so any measured difference is attributable to protection alone.
A thin adapter wraps each official model without touching its weights and reproduces its
uncompressed forward exactly, matching the reference features with a cosine similarity of
$1.000$ at $r{=}0$, so every change we report is due to reduction rather than to a
re-implementation. Positions are handled at merge time: when two patches merge, the
surviving token inherits the grid position of the patch that absorbs it, so every token
keeps a defined position throughout. This matters for rotary position embeddings
(RoPE)~\cite{rope}, which, instead of adding a position vector, rotate each token's query
and key by an angle set by its position and re-apply the rotation at every block to encode
relative position. DINOv3 and ViT-5 keep registers and patches in separate positional
spaces, so averaging a register into a patch would leave the merged token's rotation
undefined; because registers are protected and merged patches inherit a real grid
position, our reduction never mixes the two spaces and remains valid under RoPE.

\paragraph{Instantiation and Scope.}
We apply the rule identically on top of ToMe and PiToMe, altering neither
their schedules nor their hyperparameters; the few additionally retained
registers make the FLOP difference against the unprotected baselines
negligible ($<0.1\%$). For dense prediction, merged tokens are
un-merged---copied back to the positions of their constituents---to restore
the full patch grid before the readout. The rule presupposes explicit
registers. For encoders without them, protecting the top high-norm tokens is
the natural analogue, but such encoders turn out to be already robust to
merging, so we state this as a limit of scope rather than as a method.

% \section{Method}
% \label{sec:method}

% \begin{figure}[t]
% \centering
% \includegraphics[width=\linewidth]{fig_method.pdf}
% \caption{Register-aware token reduction. Each block is attention, merge, and MLP, and token tokens are merged $r$ pairs at a time by size-weighted bipartite soft matching. The protected set, the class token and the registers, is excluded from merging and passes through. The upper row ToMe keeps only the class token and merges the registers away early under extreme compression, while the lower row Ours preserves the registers throughout.}
% \label{fig:method}
% \end{figure}

% A ViT processes tokens $X=[x_1,\dots,x_T]$, with $p$ global tokens at
% the front (a class token and $p-1$ registers). Standard token merging (ToMe's
% \emph{bipartite soft matching}) splits the tokens into two sets, pairs each token in one
% set with its most similar token in the other, and merges $r$ pairs by a size-weighted
% average per block (a merged token carries the number of tokens it represents as its
% \emph{size}); it protects only the class token. PiToMe keeps this bipartite soft matching and decides \emph{what} to merge by an \emph{energy} score. Each token's energy is the mean over all other tokens of its cosine similarity minus a margin, so a token that is redundant with its neighbors has high energy. Only the highest-energy tokens are merged while the low-energy, distinctive tokens are preserved, which keeps the spectrum of the similarity graph, and the margin decreases with depth.

% \textbf{Why protect registers.} Registers are a few input-independent learned
% tokens that concentrate global rather than local/positional information
% (under linear probing they score high on classification and low on position
% recovery). In large ViTs the global representation is in fact dominated by these
% register tokens~\cite{clsregdecouple}, and their norms are so large that they act
% as attention sinks that absorb attention mass. Standard merging is
% blind to token \emph{identity} (register vs.\ token) and treats tokens homogeneously
% by similarity and size; under extreme compression it therefore merges these few
% high-norm registers into ordinary tokenes early, absorbing them into a token
% average. The tokens carrying global information then vanish, and the quality of the
% final representation collapses even though it is read only from the class token.
% We therefore exclude registers from merging explicitly.

% \textbf{Register-aware protection.} We widen the protected set $\mathcal{P}$ from
% $\{\text{CLS}\}$ (protect $1$, standard ToMe) to
% $\mathcal{P}=\{\text{CLS}\}\cup\{\text{registers}\}$ (protect $p$, i.e., the
% entire prefix; Fig.~\ref{fig:method}). This is not a new merging algorithm but a protection-rule within the ToMe merging framework, the same place where PiToMe adds an energy-based rule, deciding what to preserve. On encoders without explicit registers, protecting the top few
% high-norm tokens is a natural analogue, but such encoders are already robust to
% merging, so this is a scope limitation rather than a gain. Protected tokens are excluded from merging and passed through unchanged, and only
% the remaining token tokens are merged by the same size-weighted bipartite soft matching. The method is training-free. For
% dense prediction, merged tokens can be restored to the original resolution
% (un-merge). All comparisons use the faithful harness exactly as in the prior work
% (ToMe and PiToMe), with proportional-attention bias,
% attention-key similarity, and merging between attention and MLP, so the only
% difference among the three methods is what is protected and merged. Here
% \emph{proportional attention} is ToMe's correction that scales a token's attention weight
% by its size (the number of original tokens it represents), so a merged token is weighted
% proportionally.

\section{Experiments}
\label{sec:exp}
\subsection{Experimental Setup}
\label{subsec:setup}
 
We evaluate register protection, a rule that keeps register tokens intact during
token merging, on frozen vision transformers that contain register tokens. We
consider two tasks, image classification and semantic segmentation. Register
protection is a token-selection rule rather than a merging algorithm, so we apply
it on top of two existing merging methods, ToMe~\cite{tome} and
PiToMe~\cite{pitome}, and measure the effect of preserving the registers under a
fixed token budget. Every number we report, including the uncompressed reference
accuracy, is measured. All experiments run on a single NVIDIA A100 GPU.
 
\textbf{Models.} Register tokens are a recent addition to vision transformers, and
only a few publicly released encoders currently include them. We therefore
evaluate every register-bearing family available to us: DINOv2~\cite{dinov2},
DINOv3~\cite{dinov3}, and ViT-5~\cite{vit5}. Each backbone is taken from its
official release and kept frozen, with no fine-tuning and no additional
parameters. DINOv2 is our primary model, for which we use the ViT-B/14 and
ViT-S/14 variants. Both append four register tokens to the patch tokens and use
absolute position embeddings on a patch-14 grid. DINOv3 (S+ and B) and ViT-5 (B)
likewise use four register tokens.
 
\textbf{Datasets.} For classification we use ImageNet-1k~\cite{imagenet} with
$k$-nearest-neighbor (kNN) classification ($k{=}20$), following the DINOv2 evaluation. kNN classification attaches no trained classifier to the backbone: it labels an image by finding the most similar labeled images in feature space and taking a majority vote among them, so its accuracy reflects the quality of the frozen features themselves rather than of a head fitted on top of them. Concretely, we extract features from all training images to form a labeled reference set, and classify each validation image by a majority vote over its $k{=}20$ nearest neighbors in that set, at $224{\times}224$ resolution, reporting top-1 accuracy. For segmentation we use
ADE20k~\cite{ade20k,ade20kijcv} and report mean Intersection-over-Union (mIoU). We train a single linear head once on the frozen uncompressed features and reuse it for every method, so neither the backbone nor the merging rule is trained. The head is trained on all $20{,}210$ ADE20k training images and evaluated on the full $2{,}000$-image validation set with a $16{\times}16$ token grid.
 
\subsection{Image Classification}
 
We evaluate register protection in image classification using the ImageNet-1k with DINOv2 ViT-B/14. Register protection is applied to ToMe and compared against the ToMe and PiToMe baselines, which protect only the class token, across a range of token-reduction ratios. Table ~ref{tab:main} reports top-1 accuracy. Register protection improves accuracy at every ratio, with the improvement growing with compression, from $+0.89$ at 37$\%$ reduction to $+7.28$ at $92\%$. At the most aggressive setting, our method retains $77.28\%$, only $3.6$ points below the uncompressed model, whereas the baseline drops to $70.00\%$, $10.9$ points below. Because protecting the registers keeps four tokens that ToMe would otherwise merge, the two methods differ by exactly those four tokens at the tightest budget. Following prior work, each per-ratio entry uses a single seed, so we attach no significance test to the individual entries; run-to-run variation is quantified separately through bootstrap confidence intervals on the register-count.
sweep.
 
\begin{table}[t]
\centering
\caption{ImageNet-1k accuracy (\%) on DINOv2 ViT-B/14 with four registers. The
uncompressed model reaches $80.87$. ToMe protects only the class token. Ours additionally protects the register tokens, so at the tightest budget Ours keeps exactly the four protected registers more than
ToMe.}
\label{tab:main}
\begin{tabular}{lccc}
\toprule
Token reduction & ToMe & Ours & $\Delta$ \\
\midrule
37\% & 79.64 & \textbf{80.53} & $+0.89$ \\
55\% & 78.52 & \textbf{80.15} & $+1.63$ \\
74\% & 75.99 & \textbf{79.41} & $+3.42$ \\
83\% & 73.79 & \textbf{78.67} & $+4.88$ \\
92\% & 70.00 & \textbf{77.28} & $+7.28$ \\
\bottomrule
\end{tabular}
\end{table}
 
What the Gain Comes From. Two controls isolate the source of the improvement. First, on a DINOv2 without registers, where the protected set reduces to the class token alone, our rule and ToMe or PiToMe perform the same computation. the difference between them vanishes, and run-to-run noise stays below $0.05$ points. The gain therefore cannot appear without registers. Whether it appears specifically because of the registers is settled by the second control, in which we fix the merging and vary only which tokens are protected, always protecting the same number of tokens in addition to the class token (Table~\ref{tab:ablation}). At $92\%$ reduction, protecting a random set gives $69.53\%$, a set chosen by mean similarity $70.19\%$, and a set chosen by high norm $70.20\%$, all close to the unprotected baseline of $70.00\%$. Protecting the registers instead gives $77.28\%$, about $7$ points above these alternatives. The non-register criteria fluctuate around the baseline without a systematic gain, so we do not claim any ordering among them; only register protection clearly separates from the baseline.
 
\begin{table}[t]
\centering
\caption{Effect of the protected set. The merging is held fixed and only the
protected tokens change, with each variant protecting the same number of tokens in
addition to the class token. Only register protection clearly exceeds the
unprotected baseline, while the random, mean-similarity, and high-norm variants
stay close to it. The energy column here is a mean-similarity proxy and is
distinct from PiToMe. DINOv2 with registers.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Reduction & ToMe & \textbf{Ours} & Random & Energy & High-norm \\
\midrule
37\% & 79.64 & \textbf{80.53} & 79.71 & 79.67 & 79.67 \\
55\% & 78.52 & \textbf{80.15} & 78.45 & 78.64 & 78.56 \\
74\% & 75.99 & \textbf{79.41} & 76.05 & 76.17 & 76.29 \\
83\% & 73.79 & \textbf{78.67} & 73.83 & 74.02 & 74.11 \\
92\% & 70.00 & \textbf{77.28} & 69.53 & 70.19 & 70.20 \\
\bottomrule
\end{tabular}
\end{table}
 
One caveat applies to Table~\ref{tab:ablation}. The random, mean-similarity, and
high-norm sets are selected once from the input-layer embeddings and then held
fixed, yet the high-norm outlier tokens only emerge in the middle of the network.
In a supplementary check on the full $50{,}000$-image validation set (base model,
kNN, controlled harness), reselecting the high-norm or energy set at every block,
rather than once at the input, strengthens these baselines. At $91\%$ reduction the
best dynamic criterion reaches $67.4\%$, compared with about $64\%$ for the static
criteria and the unprotected baseline, and $71.9\%$ for register protection. These
controlled-harness absolutes are not directly comparable to the faithful values in
Table~\ref{tab:ablation}; what matters is the gap within each harness. Dynamic
reselection thus closes about half of the gap, but register protection still leads
by roughly $4.4$ points. The conclusion that only registers help holds strictly
for the static criteria of Table~\ref{tab:ablation}, and register protection
remains ahead under extreme compression even against dynamically reselected
criteria.

\subsection{Compare to baseline}

To understand the mechanism, we trace merging directly in the unprotected
baseline. On DINOv2 with registers at $r{=}16$ ($74\%$ reduction), averaged over
$16$ images, $94\%$ of the register tokens are merged into other tokens; the first
such merge occurs on average at the $3.6$th of the $12$ blocks, as early as block
$0$, and only $6\%$ of the registers survive to the final layer. Protecting the
registers keeps all of them by construction. This supports the interpretation that
the gain comes from preventing the early destruction of the register tokens rather
than from an indirect effect, although the measurement uses a small image sample
and a single value of $r$. The same behavior appears in the attention maps
(Fig.~\ref{fig:attention}): in the uncompressed model a large share of the class
token's attention falls on the few register tokens, forming an attention sink,
whereas unprotected merging folds these registers into other tokens and disperses
this attention pattern.
 
\subsubsection{Merging Methods}
 
The results so far apply register protection on top of ToMe. To test whether the
effect extends beyond ToMe, we add the same register protection on top of the
official PiToMe merging, denoted PiToMe$+$reg. PiToMe extends the same bipartite
soft matching as ToMe with an energy criterion, and register protection is a
selection rule that operates within the same framework, so the two are composable
rather than mutually exclusive. Table~\ref{tab:generality} shows that protecting
the registers raises accuracy at every reduction ratio on PiToMe as well, with a
positive gain throughout that grows with compression and reaches $+5.07$ at $92\%$.
Register protection therefore behaves as a general rule across both merging
methods we test, not a ToMe-specific effect. This matches our positioning: in
parallel with the way PiToMe adds an energy rule to the merging framework, we add
a register-protection rule, and the two can be applied together.
 
\begin{table}[t]
\centering
\caption{Register protection on top of the official PiToMe merging (PiToMe$+$reg).
DINOv2 ViT-B/14 with four registers, standard kNN ($k{=}20$), uncompressed
accuracy $80.87$. The reg gain is PiToMe$+$reg minus PiToMe, the additional
accuracy that register protection contributes on PiToMe, and it is positive at
every ratio.}
\label{tab:generality}
\begin{tabular}{lccc}
\toprule
Token reduction & PiToMe & \textbf{PiToMe$+$reg} & reg gain \\
\midrule
37\% & 79.89 & \textbf{80.37} & $+0.48$ \\
55\% & 79.16 & \textbf{80.11} & $+0.94$ \\
74\% & 76.94 & \textbf{79.06} & $+2.13$ \\
83\% & 74.73 & \textbf{78.13} & $+3.40$ \\
92\% & 71.08 & \textbf{76.15} & $+5.07$ \\
\bottomrule
\end{tabular}
\end{table}
 
\subsubsection{Encoder Families}
 
Our experiments so far use DINOv2. To test whether register protection is specific
to this model or holds for other register-bearing encoders, we evaluate
DINOv3-S+/B~\cite{dinov3} and ViT-5~\cite{vit5}, taken directly from their official
releases, under the same standard kNN protocol. All three use four registers but,
unlike DINOv2, use rotary position embeddings (RoPE)~\cite{rope}. With RoPE,
merging a register into a token breaks positional alignment, so a drop in accuracy
cannot be attributed cleanly to lost registers rather than to RoPE misalignment.
We therefore first confirm that our adapter reproduces the official forward pass
exactly at no compression, reaching a cosine similarity of $1.000$ at $r{=}0$. We
then compare register-protecting reduction (Ours) against the same backbone with
the registers removed (no-reg), which isolates the contribution of the registers
in a way that is safe under RoPE.
 
As shown in Table~\ref{tab:extra}, on all three models Ours preserves accuracy far
into extreme compression, whereas removing the registers collapses it. DINOv3 is
so dependent on its registers that removal drops it to near-chance already at
moderate compression: no-reg falls to $19.56$ at $48\%$ and $7.89$ at $95\%$ for
DINOv3-S+, and to $14.75$ and $5.60$ for DINOv3-B. ViT-5 degrades more gracefully,
from $71.17$ to $61.43$. When compression is pushed to $97\%$, leaving at most one
token, Ours also declines, to $54.8$ for DINOv3-B and $70.7$ for ViT-5, but it
stays far above the register-free baseline of $1.8$ and $42.3$. At this extreme,
sweeping the number of protected registers from $0$ to $4$ raises accuracy
monotonically, from $1.8$ to $54.8$ for DINOv3-B and from $42.3$ to $70.7$ for
ViT-5, which again attributes the gain to register protection. Treating the
registers as a protection prior is therefore not specific to DINOv2 and
generalizes to RoPE-based register encoders.
 
\begin{table}[t]
\centering
\caption{Extension to other encoder families. Standard kNN top-1 accuracy. Ours
protects the registers and then merges the remaining tokens; no-reg removes the
registers from the same backbone. DINOv3-S+/B and ViT-5 use RoPE and a patch-16
grid, so block-wise removal at $r{=}8/12/16$ corresponds to about $48/72/95\%$
reduction. DINOv2-S/B use a patch-14 grid, so their reduction points are about
$55/74/92\%$, and the DINOv2-B no-reg row is a separately trained register-free
DINOv2. Our adapter matches the official forward pass at $r{=}0$ with cosine
similarity $1.000$.}
\label{tab:extra}
\begin{tabular}{llcccc}
\toprule
Model & Method & Uncompr. & ${\sim}48\%$ & ${\sim}72\%$ & ${\sim}95\%$ \\
\midrule
DINOv2-B  & \textbf{Ours} & 80.87 & \textbf{80.15} & \textbf{79.41} & \textbf{77.28} \\
          & no-reg        & 75.85 & 74.57          & 73.73          & 71.68          \\
\midrule
DINOv2-S  & \textbf{Ours} & 77.41 & \textbf{75.80} & \textbf{74.23} & \textbf{69.85} \\
\midrule
DINOv3-S+ & \textbf{Ours} & 77.94 & \textbf{77.37} & \textbf{75.91} & \textbf{70.32} \\
          & no-reg        &       & 19.56          & 14.16          & 7.89           \\
\midrule
DINOv3-B  & \textbf{Ours} & 81.63 & \textbf{81.16} & \textbf{80.04} & \textbf{75.09} \\
          & no-reg        &       & 14.75          & 10.95          & 5.60           \\
\midrule
ViT-5-B   & \textbf{Ours} & 82.40 & \textbf{81.75} & \textbf{80.84} & \textbf{78.77} \\
          & no-reg        &       & 71.17          & 68.36          & 61.43          \\
\bottomrule
\end{tabular}
\end{table}
 
\subsection{Semantic Segmentation}
 
We now turn from classification to dense prediction, where the model assigns a
class to every token rather than a single label to the whole image. Spatial
structure matters in this setting, so preserving the registers, which aggregate
global information, is expected to matter even more than in classification. As
described in Section~\ref{subsec:setup}, we train one linear probe on the frozen
uncompressed features and reuse it for every method, so no method benefits from a
specialized head; at inference we unmerge the tokens back onto the token grid and
measure mIoU under token reduction, with neither the backbone nor our rule trained.
 
Because the $16{\times}16$ grid is coarse, the absolute mIoU is low, so the
comparison is best read in relative terms. Table~\ref{tab:dense} reports the
results. On DINOv2 with registers, register protection raises mIoU at every
reduction ratio, and the gain grows with compression, reaching $+5.7$ at $91\%$
($25.29$ versus $19.59$). In contrast to the classification experiment, the
random, energy, and high-norm variants, each protecting the same number of tokens,
barely move from the unprotected baseline and drop to or below it at the highest
reduction, whereas register protection leads by a wide margin at every ratio. The
register advantage is therefore even cleaner in this dense setting.
 
\begin{table}[t]
\centering
\caption{ADE20k linear-probe segmentation mIoU (\%) under token reduction. DINOv2
ViT-B/14 with four registers. The probe is trained on all $20{,}210$ training
images and evaluated on the full $2{,}000$-image validation set with a
$16{\times}16$ grid, so the absolute mIoU is low. Uncompressed model: $29.40$. Only
register protection exceeds the unprotected baseline at every reduction ratio.}
\label{tab:dense}
\begin{tabular}{lccccc}
\toprule
Reduction & ToMe & \textbf{Ours} & Random & Energy & High-norm \\
\midrule
37\% & 28.50 & \textbf{29.00} & 28.50 & 28.52 & 28.57 \\
55\% & 27.06 & \textbf{28.85} & 27.42 & 27.46 & 27.55 \\
74\% & 24.96 & \textbf{27.83} & 25.15 & 25.13 & 25.34 \\
83\% & 22.96 & \textbf{27.10} & 23.37 & 23.37 & 23.45 \\
91\% & 19.59 & \textbf{25.29} & 18.98 & 19.29 & 19.54 \\
\bottomrule
\end{tabular}
\end{table}
 

 
\subsection{Overhead}
 
Register protection changes which tokens are kept, not how many, so it adds almost
no compute over an unprotected merger. Token reduction lowers the backbone FLOPs by
$17\%$ to $43\%$ across our range, reaching $43\%$ at $92\%$ token reduction. This
saving is sub-linear in the token count because tokens are removed progressively,
so the early blocks still process many tokens, and because the projection and MLP
FLOPs dominate. The token-count reduction therefore overstates the FLOP saving:
$92\%$ fewer tokens corresponds to a $43\%$ FLOP reduction (Table~\ref{tab:pitome}).
Our range of $17\%$ to $43\%$ is comparable, on the same FLOP scale, to the $40\%$
to $60\%$ reported by PiToMe, since we operate in a lower-budget regime. The FLOPs
of Ours and the unprotected baseline agree to within $0.1\%$, so the extra retained
registers are negligible and the accuracy gain does not come from additional
compute.
 
At the same budget, Ours also reaches higher accuracy than PiToMe at every
reduction ratio, and the margin grows with compression, from $+0.64$ at $37\%$ to
$+6.20$ at $92\%$ (Table~\ref{tab:pitome}). Speed is likewise unaffected:
Table~\ref{tab:throughput} shows that GPU throughput is nearly identical across the
three methods and rises with compression, reaching about $1.6\times$ the
uncompressed throughput at $92\%$, roughly $570$ images per second. Register
protection therefore adds no speed cost.
 
We do not claim a better merging algorithm, but a rule that treats the registers as
an explicit prior for which tokens to keep. Like the energy rule of PiToMe, it
operates within the standard token-merging framework, and it keeps the accuracy
drop small under extreme compression without added compute or any loss of speed.
 
\begin{table}[t]
\centering
\caption{Same-budget comparison with the official PiToMe. DINOv2 ViT-B/14 with
four registers, standard kNN ($k{=}20$), uncompressed accuracy $80.87$. All three
methods are re-measured in our reproduction. Best in bold, second best underlined.
FLOP savings is the actual compute saving of progressive merging, which is smaller
than the token-count reduction (see text). GFLOPs is the per-image compute
(uncompressed $23.5$), counted following standard practice (\texttt{fvcore}
multiply-accumulate counting) as in ToMe and PiToMe.}
\label{tab:pitome}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lllcccc}
\toprule
Token reduction & FLOP savings & GFLOPs\,$\downarrow$ & ToMe & PiToMe & \textbf{Ours} & $\Delta$(O$-$P) \\
\midrule
37\% & 17\% & 19.4 & 79.64 & \underline{79.89} & \textbf{80.53} & $+0.64$ \\
55\% & 26\% & 17.4 & 78.52 & \underline{79.16} & \textbf{80.15} & $+0.99$ \\
74\% & 35\% & 15.4 & 75.99 & \underline{76.94} & \textbf{79.41} & $+2.47$ \\
83\% & 39\% & 14.4 & 73.79 & \underline{74.73} & \textbf{78.67} & $+3.94$ \\
92\% & 43\% & 13.4 & 70.00 & \underline{71.08} & \textbf{77.28} & $+6.20$ \\
\bottomrule
\end{tabular}}
\end{table}
 
\begin{table}[!h]
\centering
\caption{GPU throughput (images per second) versus token reduction, DINOv2 ViT-B/14.
The three methods nearly overlap, so protecting the registers does not reduce
speed.}
\label{tab:throughput}  
\begin{tabular}{lccc}
\toprule
Token reduction & ToMe & PiToMe & \textbf{Ours} \\
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


\section{Discussion and Limitation}
\label{sec:limitations}
\paragraph{Encoders without registers.} On DINOv2 without registers, no keep-prior (including high-norm protection) exceeds the baseline (all within ${\sim}0.2$ points), and that model is already robust to merging (losing only ${\sim}4$ points at $93\%$). We therefore restrict our claim to register-equipped encoders and do \emph{not} claim the high-norm analogue as a working method.

\paragraph{Model dependence.} We report gains under two kNN protocols. The train-gallery setup uses the full ImageNet training set as the reference gallery; val leave-one-out (val-LOO) instead uses the $50{,}000$-image validation set as both the gallery and the query, classifying each image by the nearest neighbors among the \emph{other} validation images with itself excluded, so it cannot retrieve its own label. The smaller val-LOO gallery yields lower absolute accuracy but preserves the ordering of methods. Across the three register models the size of the gain varies. On DINOv2-ViT-B/14, register protection clearly leads under extreme reduction (val-LOO $+10.2$ at $91\%$; train $+7.3$). On DINOv2-ViT-L/14, the unprotected baseline is far more fragile to merging (collapsing to $2.8\%$ at $91\%$), so register protection has a large relative gain ($+5.8$) even though its absolute value is low ($8.6\%$ vs $2.8\%$ unprotected). On the smaller DINOv2-ViT-S/14, the unprotected baseline is already robust to merging, so the gain is small ($+0.9$ at $91\%$) and a dynamic energy criterion slightly outperforms register protection ($61.5$ vs $60.7$). We therefore characterize the effect as model-dependent, largest in the extreme regime where registers become the bottleneck.

\paragraph{Scale and uncertainty.} Merging and kNN are deterministic, but feature extraction has ${\sim}0.1$-point variation across sub-experiments from GPU nondeterminism. Instead of seed repetition we show significance with bootstrap CIs, which exclude $0$ at every reduction ratio, including moderate compression.


\section{Conclusion}
\label{sec:conclusion}

In this work, we introduced register-aware token reduction, a training-free plug-in that places register tokens in the protected set during merging, so that only patches are merged while the registers are kept. Because registers are trained to absorb global information, keeping them preserves what ordinary merging quietly discards. On a register-equipped DINOv2-ViT-B/14 evaluated with the standard train-dataset kNN, our method outperforms ToMe-style baselines at every reduction ratio, and the gap widens under stronger compression, reaching $+7.3$ percentage points in top-1 accuracy at ${\sim}92\%$ token reduction. An equal-count ablation isolates this gain to the registers themselves rather than to protecting more tokens.

The protection rule is not specific to ToMe. The same protection helps on top of official PiToMe's merging, and it generalizes to other register-bearing encoders such as DINOv3 and ViT-5, where removing the registers collapses accuracy under extreme compression and confirms that registers are the bottleneck. The gain comes at no extra compute, so throughput is essentially identical across methods, and a linear probe, retrieval mAP, bootstrap confidence intervals, the register-count sweep, exhaustive val leave-one-out, weighted voting, and a ViT-S reproduction all point the same way. The gain is, however, model-dependent, clear on base and large but weak on small, and largest in the extreme-compression regime where registers become the bottleneck. Register protection is thus a lightweight, general prior for any register-bearing encoder, and we hope it encourages token-reduction methods to preserve registers explicitly.

% ---------------------------------------------------------------
\bibliographystyle{splncs04}
\bibliography{main}

\end{document}
