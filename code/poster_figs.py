"""포스터(poster.html)에 넣을 그래프 3개와 QR 코드 2개를 SVG로 만들어 끼워 넣는 스크립트.

저장소 맨 위 폴더에서 실행한다: python3 code/poster_figs.py
check.py와 같은 전처리·분할·모델을 사용하므로 그래프의 값은 check.py 수치와 같다.
포스터를 만들 때만 사용하는 도구라 matplotlib·segno·xgboost가 필요하다(앱 실행에는 필요 없음).
  pip install matplotlib segno xgboost

poster.html의 <!-- FIG:이름 --> 과 <!-- /FIG:이름 --> 사이를 새 SVG로 바꾼다.
  FIG:hist     2. 데이터  분포 그래프
  FIG:bars     4. 결과    평가지표 비교 막대 그래프(기준선 + 후보 모델 전부, 선택 모델만 강조 색)
  FIG:result   4. 결과    결과 그래프(회귀: 실제-예측 산점도)
  FIG:qr_repo  제목띠     저장소 QR
  FIG:qr_app   발치       앱 QR

[다른 샘플에서 이 파일을 쓸 때]
  · 공통 부분(설정·svg_of·qr_svg·inject)은 그대로 둔다.
  · "이 샘플의 데이터와 그래프" 부분만 바꾼다. 그래프 이름(FIG:이름) 다섯 개는 그대로 사용한다.
  · 그래프 폭은 COL_W(단 폭)로 고정한다. 폭을 맞춰야 그래프 글자 20pt가 포스터에서도 20pt로 보인다.
"""
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 끄기
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import segno

# ───────────── 공통 설정(다섯 샘플이 같다) ─────────────
ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster.html"
ACCENT = "#1f3d7a"      # poster.html의 --accent와 같은 남색
GRAY = "#9aa0aa"        # 선택하지 않은 모델·나머지 점
DARK = "#4a4f57"        # 기준선
# 단 폭(인치) = (594mm - 좌우 여백 2×26mm - 단 사이 22mm) / 2 = 260mm. poster.html의 --pad-x·--gap을 바꾸면 다시 계산한다
COL_W = 260 / 25.4

plt.rcParams.update({
    "svg.fonttype": "none",          # 글자를 글자로 남겨 브라우저 글꼴로 그린다
    "font.family": ["Apple SD Gothic Neo", "Noto Sans KR", "sans-serif"],
    "font.size": 20,                 # 포스터의 표·그림 설명 20pt와 맞춘다
    "axes.edgecolor": "#000", "axes.labelcolor": "#000",
    "xtick.color": "#000", "ytick.color": "#000",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "svg.hashsalt": "poster",        # 다시 만들어도 같은 SVG가 나오게 한다
})


def svg_of(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]                       # XML 머리말·DOCTYPE 제거
    s = re.sub(r'\swidth="[^"]+"\sheight="[^"]+"', "", s, count=1)  # 크기는 CSS가 정한다
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    return s


def qr_svg(url):
    qr = segno.make(url, error="m")
    n = qr.symbol_size(scale=1, border=2)[0]
    buf = io.BytesIO()
    qr.save(buf, kind="svg", xmldecl=False, svgns=True, scale=1, border=2, dark="#000")
    s = buf.getvalue().decode()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', f' viewBox="0 0 {n} {n}" shape-rendering="crispEdges"', s, count=1)
    return s.replace("<svg ", f'<svg role="img" aria-label="{url}" ', 1)


def inject(figs):
    html = POSTER.read_text(encoding="utf-8")
    for name, svg in figs.items():
        pat = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.S)
        if not pat.search(html):
            raise SystemExit(f"poster.html에 FIG:{name} 표시가 없습니다")
        html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
    POSTER.write_text(html, encoding="utf-8")


def bar_chart(names, values, pick, base, fmt, xlabel, height=4.6):
    """평가지표 비교 가로 막대. pick=선택 모델 이름, base=기준선 이름."""
    fig, ax = plt.subplots(figsize=(COL_W, height))
    y = np.arange(len(names))[::-1]
    colors = [ACCENT if n == pick else DARK if n == base else GRAY for n in names]
    ax.barh(y, values, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t, n in zip(ax.get_yticklabels(), names):
        if n == pick:
            t.set_color(ACCENT)
            t.set_fontweight("bold")
    vmax = max(values)
    for yi, v, n in zip(y, values, names):
        ax.text(v + vmax * 0.012, yi, fmt(v), va="center", fontsize=20,
                color=ACCENT if n == pick else "#000", fontweight="bold" if n == pick else "normal")
    ax.set_xlim(0, vmax * 1.16)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}" if t >= 1 or t == 0 else f"{t:.2f}"))
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return svg_of(fig)


# ───────────── 이 샘플의 데이터와 그래프(샘플마다 바꾼다) ─────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

DATA = ROOT / "data" / "diabetes.csv"
REPO_URL = "https://github.com/greatsong/ds-sample-5-biochem"
APP_URL = "https://greatsong.github.io/ds-sample-5-biochem/"
LIGHT = {ACCENT: "#b4c1db", DARK: "#c4c7cc", GRAY: "#dcdfe3"}   # 훈련 MAE 막대(같은 색의 옅은 판)

# check.py와 같은 데이터 준비(최종 변수 묶음 bmi + ltg, 같은 분할)
df = pd.read_csv(DATA)
train, test = train_test_split(df, test_size=0.2, random_state=42)
prog = df["progression"]
y = test["progression"]
X = ["bmi", "ltg"]

# 그래프 1(FIG:hist): 진행 점수 분포(442명)
fig, ax = plt.subplots(figsize=(COL_W, 2.0))
ax.hist(prog, bins=np.arange(20, 361, 20), color=ACCENT, edgecolor="#fff", linewidth=0.6)
top = ax.get_ylim()[1]
for val, lab, dy in [(prog.median(), f"중앙값 {prog.median():.1f}점", 0.55), (prog.mean(), f"평균 {prog.mean():.1f}점", 0.85)]:
    ax.axvline(val, color="#000", linestyle=(0, (4, 3)), linewidth=1.4)
    # 두 선이 가까워 글자는 오른쪽에 두고 가는 선으로 잇는다
    ax.annotate(lab, xy=(val, top * dy), xytext=(250, top * dy), va="center", fontsize=20,
                arrowprops=dict(arrowstyle="-", color="#000", linewidth=1.0, shrinkA=4, shrinkB=0))
ax.set_xlabel("1년 뒤 진행 점수(점)")
ax.set_ylabel("환자 수(명)", labelpad=14)
ax.set_xlim(0, 360)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
svg_hist = svg_of(fig)

# 그래프 2(FIG:bars): 기준선과 후보 모델 다섯 개의 시험 MAE(진한 막대)와 훈련 MAE(옅은 막대)
base_pred = train["progression"].mean()
rows = [("평균 모델(기준선)", mean_absolute_error(y, np.full(len(test), base_pred)),
         mean_absolute_error(train["progression"], np.full(len(train), base_pred)))]
models = [
    ("선형 회귀(선택)", LinearRegression()),
    ("k-NN 회귀(k=5)", KNeighborsRegressor(n_neighbors=5)),
    ("결정 트리 회귀(깊이 4)", DecisionTreeRegressor(max_depth=4, random_state=0)),
    ("랜덤 포레스트 회귀", RandomForestRegressor(n_estimators=300, random_state=0)),
    ("XGBoost 회귀", XGBRegressor(random_state=0)),
]
for name, m in models:
    m.fit(train[X], train["progression"])
    rows.append((name, mean_absolute_error(y, m.predict(test[X])),
                 mean_absolute_error(train["progression"], m.predict(train[X]))))
PICK, BASE = "선형 회귀(선택)", "평균 모델(기준선)"
fig, ax = plt.subplots(figsize=(COL_W, 5.1))
pos = np.arange(len(rows))[::-1] * 1.0
h = 0.42
for yi, (name, te_mae, tr_mae) in zip(pos, rows):
    c = ACCENT if name == PICK else DARK if name == BASE else GRAY
    ax.barh(yi + h / 2, te_mae, height=h, color=c)
    ax.barh(yi - h / 2, tr_mae, height=h, color=LIGHT[c])
    bold = "bold" if name == PICK else "normal"
    ax.text(te_mae + 0.8, yi + h / 2, f"{te_mae:.1f}", va="center", fontsize=20,
            color=ACCENT if name == PICK else "#000", fontweight=bold)
    ax.text(tr_mae + 0.8, yi - h / 2, f"{tr_mae:.1f}", va="center", fontsize=20, color="#555")
ax.set_yticks(pos)
ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r[0] == PICK:
        t.set_color(ACCENT)
        t.set_fontweight("bold")
ax.set_xlim(0, 100)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xlabel("MAE(점, 작을수록 좋음)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRAY, label="시험 데이터 89명"), Patch(color=LIGHT[GRAY], label="훈련 데이터 353명")],
          loc="center right", frameon=False, fontsize=20, handlelength=1.2, borderaxespad=0.1)
svg_bars = svg_of(fig)

# 그래프 3(FIG:result): 시험 데이터 89명의 실제 진행 점수와 최종 모델(선형 회귀, bmi + ltg) 예측
mc = LinearRegression().fit(train[X], train["progression"])
pred = mc.predict(test[X])
real = y.to_numpy()
high = real >= 250
fig, ax = plt.subplots(figsize=(COL_W, 4.0))
lo, hi = 0, 360
ax.plot([lo, hi], [lo, hi], color="#000", linewidth=1.4, linestyle=(0, (4, 3)))
ax.scatter(real[~high], pred[~high], s=70, color=GRAY, edgecolor="none", label=f"실제 250점 미만 {(~high).sum()}명")
ax.scatter(real[high], pred[high], s=70, color=ACCENT, edgecolor="none", label=f"실제 250점 이상 {high.sum()}명")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xticks([0, 100, 200, 300])
ax.set_yticks([0, 100, 200, 300])
ax.set_xlabel("실제 진행 점수(점)")
ax.set_ylabel("예측 진행 점수(점)", labelpad=14)
ax.text(10, hi * 0.95, "점선 위: 실제보다 높게 예측", va="top", fontsize=20)
ax.legend(loc="lower right", frameon=False, fontsize=20, handletextpad=0.2, borderaxespad=0.1)
svg_result = svg_of(fig)

inject({"hist": svg_hist, "bars": svg_bars, "result": svg_result,
        "qr_repo": qr_svg(REPO_URL), "qr_app": qr_svg(APP_URL)})
print("시험 MAE / 훈련 MAE(점):", ", ".join(f"{n} {a:.1f}/{b:.1f}" for n, a, b in rows))
print(f"poster.html 갱신: 그래프 3개, QR 2개 (시험 데이터 {len(test)}명, 실제 250점 이상 {high.sum()}명 중 "
      f"낮게 예측 {(pred[high] < real[high]).sum()}명)")
