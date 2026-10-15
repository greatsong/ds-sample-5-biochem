"""README·log·slides에 적은 수치를 다시 계산하는 검증 스크립트.

저장소 맨 위 폴더에서 실행한다: python3 code/check.py
앱(main.py)과 같은 분할·모델을 사용한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

try:  # XGBoost는 로컬에서만 계산한다(브라우저 앱에는 넣을 수 없다)
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

DATA = Path("data/diabetes.csv")
if not DATA.exists():
    DATA = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"

ALL = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
SETS = {
    "bmi 하나": ["bmi"],
    "bmi + ltg": ["bmi", "ltg"],
    "열 개 전부": ALL,
}

df = pd.read_csv(DATA)
y_all = df["progression"]
print("== 1. 데이터 ==")
print(f"전체 행 수: {len(df)}명, 열 수: {df.shape[1]}개")
print(f"결측값: {int(df.isna().sum().sum())}개")
sc = df["sex"].value_counts().sort_index()
print(f"sex 값: 1이 {sc[1.0]}명, 2가 {sc[2.0]}명")
print(f"progression 평균 {y_all.mean():.1f}, 중앙값 {y_all.median():.1f}, 최솟값 {y_all.min():.0f}, 최댓값 {y_all.max():.0f}, 표준편차 {y_all.std():.1f}")
print(f"progression 250 이상: {(y_all >= 250).sum()}명 ({(y_all >= 250).mean() * 100:.1f}%)")
print(f"왜도(치우친 정도): {y_all.skew():.2f}")
print(f"나이 범위: {df['age'].min():.0f}~{df['age'].max():.0f}살, bmi 범위: {df['bmi'].min():.1f}~{df['bmi'].max():.1f}")
print(f"ltg 범위: {df['ltg'].min():.2f}~{df['ltg'].max():.2f}")
print(f"ltg를 자연로그로 보고 되돌린 값: {np.exp(df['ltg'].min()):.0f}~{np.exp(df['ltg'].max()):.0f}, 중앙값 {np.exp(df['ltg'].median()):.0f}")
print(f"ltg를 밑 10 로그로 보고 되돌린 최댓값: {10 ** df['ltg'].max():,.0f}")
print("열 단위 확인(평균): " + ", ".join(f"{c} {df[c].mean():.1f}" for c in ["bp", "tc", "ldl", "hdl", "glu"]))
ratio = df["tc"] / df["hdl"]
print(f"tch 값 중 정수 비율: {(df['tch'] == df['tch'].round()).mean() * 100:.0f}%, tc÷hdl과의 상관계수: {ratio.corr(df['tch']):.2f}")
g = df.groupby("sex")[["hdl", "progression"]].mean()
print(f"sex별 평균 hdl: 1은 {g.loc[1.0, 'hdl']:.1f}, 2는 {g.loc[2.0, 'hdl']:.1f} / 평균 progression: 1은 {g.loc[1.0, 'progression']:.1f}, 2는 {g.loc[2.0, 'progression']:.1f} (차이 {g.loc[2.0, 'progression'] - g.loc[1.0, 'progression']:.1f})")
print(f"tc 평균 {df['tc'].mean():.1f} = ldl {df['ldl'].mean():.1f} + hdl {df['hdl'].mean():.1f} + 나머지 {df['tc'].mean() - df['ldl'].mean() - df['hdl'].mean():.1f}")

print("\n== 2. progression과의 상관계수 (전체 442명) ==")
corr = df.corr()["progression"].drop("progression").sort_values(key=lambda s: -s.abs())
for c, r in corr.items():
    print(f"  {c}: {r:+.2f}")
print(f"tc와 ldl의 상관계수: {df['tc'].corr(df['ldl']):.2f}, tch와 hdl: {df['tch'].corr(df['hdl']):.2f}")

train, test = train_test_split(df, test_size=0.2, random_state=42)
y = test["progression"]
print(f"\n훈련 데이터 {len(train)}명, 시험 데이터 {len(test)}명")

print("\n== 3. 기준선(평균 모델) ==")
base = np.full(len(test), train["progression"].mean())
base_mae = mean_absolute_error(y, base)
print(f"훈련 데이터 평균: {train['progression'].mean():.1f}")
print(f"평균 모델 MAE: {base_mae:.1f}, R²: {r2_score(y, base):.3f}")

print("\n== 4. 변수 묶음 세 가지 ==")
models = {}
for name, cols in SETS.items():
    m = LinearRegression().fit(train[cols], train["progression"])
    p = m.predict(test[cols])
    models[name] = (m, p)
    mae = mean_absolute_error(y, p)
    print(f"{name}: R² {r2_score(y, p):.3f}, MAE {mae:.1f}, 기준선보다 MAE {(1 - mae / base_mae) * 100:.0f}% 감소")
r_two = r2_score(y, models["bmi + ltg"][1])
r_all = r2_score(y, models["열 개 전부"][1])
print(f"열 개 전부와 bmi + ltg의 R² 차이: {r_all - r_two:.4f} (소수 넷째 자리까지: {r_two:.4f}, {r_all:.4f})")
print(f"MAE 차이: {mean_absolute_error(y, models['bmi + ltg'][1]) - mean_absolute_error(y, models['열 개 전부'][1]):.1f}점")

m1 = models["bmi 하나"][0]
print(f"bmi 하나 모델 계수: bmi 1 오르면 {m1.coef_[0]:.1f}점")
m2 = models["bmi + ltg"][0]
print(f"bmi + ltg 모델 계수: bmi {m2.coef_[0]:.1f}, ltg {m2.coef_[1]:.1f}, 절편 {m2.intercept_:.1f}")

print("\n== 5. 확인용 변수 묶음 ==")
extra = {
    "bmi + ltg + bp": ["bmi", "ltg", "bp"],
    "bmi + ltg + tc + ldl": ["bmi", "ltg", "tc", "ldl"],
    "bmi + ltg + glu": ["bmi", "ltg", "glu"],
    "ltg 하나": ["ltg"],
    "sex 뺀 아홉 개": [c for c in ALL if c != "sex"],
}
for name, cols in extra.items():
    p = LinearRegression().fit(train[cols], train["progression"]).predict(test[cols])
    print(f"{name}: R² {r2_score(y, p):.3f}, MAE {mean_absolute_error(y, p):.1f}")

print("\n== 6. 열 개 전부 모델의 계수 ==")
mall = models["열 개 전부"][0]
coef = pd.Series(mall.coef_, index=ALL)
print("원래 계수(단위 1 오를 때 점수 변화): " + ", ".join(f"{c} {v:.1f}" for c, v in coef.sort_values(key=lambda s: -s.abs()).items()))
std_coef = coef * train[ALL].std()
print("표준화 계수(표준편차 1만큼 오를 때 점수 변화): " + ", ".join(f"{c} {v:.1f}" for c, v in std_coef.sort_values(key=lambda s: -s.abs()).items()))
print(f"훈련 데이터 표준편차: ltg {train['ltg'].std():.2f}, bmi {train['bmi'].std():.2f}, tc {train['tc'].std():.1f}")

print("\n== 7. 잔차 살펴보기 (열 개 전부 모델, 시험 데이터) ==")
t = test.copy()
t["pred"] = models["열 개 전부"][1]
t["err"] = t["pred"] - t["progression"]
mae_all = mean_absolute_error(y, t["pred"])
print(f"열 개 모델이 설명하지 못한 부분(1 - R²): 약 {(1 - r_all) * 100:.0f}%")
print(f"MAE {mae_all:.1f}점은 점수 범위({y_all.min():.0f}~{y_all.max():.0f}, 폭 {y_all.max() - y_all.min():.0f})의 {mae_all / (y_all.max() - y_all.min()) * 100:.0f}%")
hi = t[t["progression"] >= 250]
lo = t[t["progression"] < 100]
mid = t[(t["progression"] >= 100) & (t["progression"] < 250)]
for label, s in [("250 이상", hi), ("100~249", mid), ("100 미만", lo)]:
    print(f"실제 {label} {len(s)}명: 낮게 예측 {(s['err'] < 0).sum()}명, 평균 오차(예측-실제) {s['err'].mean():+.1f}, 평균 절대 오차 {s['err'].abs().mean():.1f}")
print(f"실제 250 이상 환자의 예측 최댓값: {hi['pred'].max():.1f}")
print(f"시험 데이터 예측값 범위: {t['pred'].min():.1f}~{t['pred'].max():.1f}, 실제값 범위: {y.min():.0f}~{y.max():.0f}")
print(f"오차가 50점 넘는 환자: {(t['err'].abs() > 50).sum()}명 / {len(t)}명")
# 예측값이 비슷한데 실제가 크게 다른 쌍
srt = t.sort_values("pred")
pairs = []
vals = srt[["pred", "progression"]].to_numpy()
for i in range(len(vals) - 1):
    pairs.append((abs(vals[i + 1][1] - vals[i][1]), vals[i], vals[i + 1]))
gap, a, b = max(pairs, key=lambda x: x[0])
print(f"예측값이 이웃한 두 환자: 예측 {a[0]:.1f}·{b[0]:.1f}, 실제 {a[1]:.0f}·{b[1]:.0f} (차이 {gap:.0f})")

print("\n== 8. 예측해 보기 (bmi + ltg 모델) ==")
for b_, l_ in [(22.0, 4.3), (26.0, 4.6), (32.0, 5.2)]:
    p = m2.predict(pd.DataFrame({"bmi": [b_], "ltg": [l_]}))[0]
    print(f"bmi {b_}, ltg {l_}: {p:.1f}점")

print("\n== 9. 후보 모델 비교 (bmi + ltg, 같은 분할) ==")
X = ["bmi", "ltg"]
candidates = [
    ("선형 회귀", LinearRegression()),
    ("k-NN 회귀(k=5)", KNeighborsRegressor(n_neighbors=5)),
    ("결정 트리 회귀(깊이 4)", DecisionTreeRegressor(max_depth=4, random_state=0)),
    ("랜덤 포레스트 회귀(300그루)", RandomForestRegressor(n_estimators=300, random_state=0)),
]
if XGBRegressor is not None:
    candidates.append(("XGBoost 회귀(기본값)", XGBRegressor(random_state=0)))
else:
    print("xgboost가 설치되어 있지 않아 XGBoost 회귀는 건너뜁니다.")
base_train_mae = mean_absolute_error(train["progression"], np.full(len(train), train["progression"].mean()))
print(f"평균 모델(기준선): 시험 MAE {base_mae:.1f}, 시험 R² {r2_score(y, base):.3f}, 훈련 MAE {base_train_mae:.1f}")
scores = []
for name, m in candidates:
    m.fit(train[X], train["progression"])
    p = m.predict(test[X])
    mae = mean_absolute_error(y, p)
    tr_mae = mean_absolute_error(train["progression"], m.predict(train[X]))
    scores.append((name, mae, r2_score(y, p)))
    print(f"{name}: 시험 MAE {mae:.1f}, 시험 R² {r2_score(y, p):.3f}, 훈련 MAE {tr_mae:.1f}, "
          f"평균 모델보다 MAE {(1 - mae / base_mae) * 100:.0f}% 감소")
best_mae = min(scores, key=lambda r: r[1])
best_r2 = max(scores, key=lambda r: r[2])
lin = scores[0]
print(f"MAE가 가장 작은 모델: {best_mae[0]} ({best_mae[1]:.2f}), 선형 회귀 {lin[1]:.2f}, 차이 {lin[1] - best_mae[1]:.2f}점")
print(f"R²가 가장 높은 모델: {best_r2[0]} ({best_r2[2]:.3f})")
