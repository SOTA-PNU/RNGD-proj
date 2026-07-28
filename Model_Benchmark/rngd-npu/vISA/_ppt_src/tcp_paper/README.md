# TCP 논문(ISCA 2024) 분석 PPT 소스

`../../TCP논문-ISCA2024-분석.pptx` (258장) 의 원자료다.

| 파일 | 내용 |
|---|---|
| `isca_fig1~15.png` · `isca_tableI/II.png` | 논문에서 잘라낸 그림 15개 + 표 2개 (300 DPI) |
| `isca_index.json` | 각 그림의 원본 쪽·좌표(pt)·픽셀 크기 |
| `sections.json` | 논문 본문을 절 단위로 분할한 텍스트 |
| `extract3.py` | 그림 자동 추출기 |
| `../tcp_content.json` | 슬라이드 내용 (10개 부) |
| `../build_tcp.py` | 조립기 |

## 원논문

TCP: A Tensor Contraction Processor for AI Workloads — Industrial Product
ISCA 2024 Industry Track · FuriosaAI 외 · <https://web.ist.utl.pt/nuno.lopes/pubs/tcp-isca24.pdf>
슬라이드: <https://www.iscaconf.org/isca2024/slides/Session%207%20-%20TCP.pdf>
관련: IEEE Micro (Hot Chips 2024 특집) <https://ieeexplore.ieee.org/document/10929037/>

## 그림 추출에서 걸렸던 것

- **표 캡션이 표 아래에 있다.** TABLE I·II 둘 다 그렇다. IEEE 관례(표 캡션은 위)와 달라서
  자동 추출이 Table I 을 Fig. 3 영역으로 잡았다. 두 표는 본문 텍스트 좌표로 직접 확정했다.
- **그림 내부 라벨은 4~6pt, 본문·캡션은 10pt.** 경계로 쓸 본문 줄은 폭 55pt 이상만 인정해야 한다.
  안 그러면 그림 속 수식 기호(`…`)를 본문으로 오인해 Fig. 9 영역이 11pt 로 잘린다.
- 그림 폭이 단(column) 을 넘으면 양단 걸침(wide)이다. 캡션 폭이 페이지의 55% 를 넘는지로 판별한다.

## 다시 만들기

```bash
cd _ppt_src && /tmp/pptenv/bin/python build_tcp.py tcp_content.json ../TCP논문-ISCA2024-분석.pptx
/tmp/pptenv/bin/python ooxml_check.py  ../TCP논문-ISCA2024-분석.pptx
/tmp/pptenv/bin/python layout_check.py ../TCP논문-ISCA2024-분석.pptx
```
