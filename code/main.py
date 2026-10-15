# 혈액 생화학 수치와 체질량지수로 1년 뒤 당뇨 진행 정도를 예측할 수 있는가
# 실행: 저장소 맨 위 폴더에서 streamlit run code/main.py
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="당뇨 진행 예측", page_icon="🩸", layout="wide")

DATA = Path("data/diabetes.csv")
if not DATA.exists():  # code 폴더 안에서 실행한 경우
    DATA = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"

ALL = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
SETS = {
    "bmi 하나": ["bmi"],
    "bmi + ltg": ["bmi", "ltg"],
    "열 개 전부": ALL,
}
LABELS = {
    "age": "나이(age)", "sex": "성별 부호(sex)", "bmi": "체질량지수(bmi)", "bp": "평균 혈압(bp)",
    "tc": "총콜레스테롤(tc)", "ldl": "LDL 콜레스테롤(ldl)", "hdl": "HDL 콜레스테롤(hdl)",
    "tch": "총콜레스테롤÷HDL(tch)", "ltg": "중성지방 로그값(ltg)", "glu": "혈당(glu)",
    "progression": "1년 뒤 진행 점수(progression)",
}


# XGBoost는 브라우저(stlite)에서 실행할 수 없어서 로컬에서 code/check.py로 계산한 값을 적어 둔다
XGB_ROW = ("XGBoost 회귀(기본값) *", 50.7, 0.196, 4.4)


@st.cache_data
def load():
    return pd.read_csv(DATA)


@st.cache_resource
def fit_models(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    y = test["progression"]
    base_pred = np.full(len(test), train["progression"].mean())
    rows = [("평균 모델(기준선)", "없음", r2_score(y, base_pred), mean_absolute_error(y, base_pred))]
    models, preds = {}, {}
    for name, cols in SETS.items():
        m = LinearRegression().fit(train[cols], train["progression"])
        p = m.predict(test[cols])
        models[name], preds[name] = m, p
        rows.append((name, ", ".join(cols) if len(cols) < 10 else "열 10개", r2_score(y, p), mean_absolute_error(y, p)))
    table = pd.DataFrame(rows, columns=["모델", "입력", "R²", "MAE(점)"])
    table["R²"] = table["R²"].map(lambda r: f"{r:.3f}")
    table["MAE(점)"] = table["MAE(점)"].map(lambda m: f"{m:.1f}")
    result = test.copy()
    for name in SETS:
        result[name] = preds[name]
    return train, test, table, models, result


@st.cache_data
def compare_models(df):
    """최종 변수 묶음(bmi + ltg)으로 후보 모델을 같은 분할에서 비교한다."""
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X, y_tr, y_te = ["bmi", "ltg"], train["progression"], test["progression"]
    base_te = np.full(len(test), y_tr.mean())
    base_tr = np.full(len(train), y_tr.mean())
    rows = [("평균 모델(기준선)", mean_absolute_error(y_te, base_te), r2_score(y_te, base_te), mean_absolute_error(y_tr, base_tr))]
    candidates = [
        ("선형 회귀(최종)", LinearRegression()),
        ("k-NN 회귀(k=5)", KNeighborsRegressor(n_neighbors=5)),
        ("결정 트리 회귀(깊이 4)", DecisionTreeRegressor(max_depth=4, random_state=0)),
        ("랜덤 포레스트 회귀(300그루)", RandomForestRegressor(n_estimators=300, random_state=0)),
    ]
    for name, m in candidates:
        m.fit(train[X], y_tr)
        p = m.predict(test[X])
        rows.append((name, mean_absolute_error(y_te, p), r2_score(y_te, p), mean_absolute_error(y_tr, m.predict(train[X]))))
    rows.append(XGB_ROW)
    out = pd.DataFrame(rows, columns=["모델", "시험 MAE(점)", "시험 R²", "훈련 MAE(점)"])
    out["시험 MAE(점)"] = out["시험 MAE(점)"].map(lambda v: f"{v:.1f}")
    out["시험 R²"] = out["시험 R²"].map(lambda v: f"{v:.3f}")
    out["훈련 MAE(점)"] = out["훈련 MAE(점)"].map(lambda v: f"{v:.1f}")
    return out


df = load()
train, test, table, models, result = fit_models(df)

st.title("🩸 혈액 생화학 수치와 체질량지수로 1년 뒤 당뇨 진행 정도를 예측할 수 있는가")
st.caption("당뇨 환자 442명 데이터 · 10101 홍길동 · 10102 김영희 · 10103 이철수 (선생님이 만든 가상 팀의 샘플)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 질문", "② 데이터 보기", "③ 기준선", "④ 모델 결과", "⑤ 해석·한계"])

with tab1:
    st.subheader("우리 팀의 질문")
    st.markdown(
        "생명과학 시간에 혈당·인슐린·지질 대사를 배웠습니다. 건강검진표의 숫자(콜레스테롤·중성지방·혈당)와 체질량지수가 "
        "**1년 뒤 당뇨가 얼마나 진행하는지를 얼마나 알려 주는지** 확인합니다. "
        "진행 정도가 숫자(25~346점)라서 회귀로 예측합니다."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("분석한 환자", f"{len(df)}명", "결측값 없음", delta_color="off")
    c2.metric("방법", "선형 회귀")
    c3.metric("평가 지표", "R² · MAE")

with tab2:
    st.subheader("1년 뒤 진행 점수의 분포")
    y_all = df["progression"]
    fig = px.histogram(df, x="progression", nbins=30, labels={"progression": "1년 뒤 진행 점수"})
    fig.update_layout(yaxis_title="환자 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평균", f"{y_all.mean():.1f}")
    c2.metric("중앙값", f"{y_all.median():.1f}")
    c3.metric("최솟값", f"{y_all.min():.0f}")
    c4.metric("최댓값", f"{y_all.max():.0f}")

    st.subheader("측정값 하나와 진행 점수")
    col = st.selectbox("가로축에 놓을 측정값", ALL, index=ALL.index("bmi"), format_func=lambda c: LABELS[c])
    fig2 = px.scatter(df, x=col, y="progression", opacity=0.6, labels=LABELS)
    st.plotly_chart(fig2, use_container_width=True)
    st.metric("진행 점수와의 상관계수", f"{df[col].corr(df['progression']):+.2f}")
    if col == "sex":
        st.warning("sex는 1과 2로 적힌 부호입니다. 어느 쪽이 남자인지 출처 문서에 없어서 뜻을 해석하지 않습니다.")
    if col == "ltg":
        st.info("ltg는 중성지방에 이미 로그를 씌운 값입니다. 다시 변환하지 않고 그대로 사용했습니다.")

    with st.expander("상관계수 전체 보기"):
        corr = df.corr()["progression"].drop("progression").sort_values(key=lambda s: -s.abs())
        st.dataframe(pd.DataFrame({"측정값": [LABELS[c] for c in corr.index], "상관계수": corr.round(2).values}),
                     hide_index=True, use_container_width=True)

with tab3:
    st.subheader("기준선: 평균 모델")
    st.markdown(
        f"측정값을 전혀 보지 않고 모든 환자의 진행 점수를 **훈련 데이터 평균({train['progression'].mean():.1f}점)**으로 예측합니다. "
        "우리 모델이 이것보다 얼마나 덜 틀리는지가 판단 기준입니다."
    )
    st.metric("평균 모델의 MAE", f"{table.loc[0, 'MAE(점)']}점")
    st.caption(
        f"훈련 데이터 {len(train)}명으로 만들고 시험 데이터 {len(test)}명으로 평가했습니다(8:2 분할). "
        "평균 모델의 R²가 0보다 조금 작은 것은 훈련 데이터 평균과 시험 데이터 평균이 조금 다르기 때문입니다."
    )

with tab4:
    st.subheader("후보 모델 비교 (bmi + ltg, 같은 분할)")
    st.dataframe(compare_models(df), hide_index=True, use_container_width=True)
    st.caption(
        "* XGBoost는 브라우저 앱에서 실행할 수 없어서 로컬에서 계산한 값입니다(code/check.py). 나머지는 앱에서 바로 계산합니다. "
        "MAE가 가장 작은 랜덤 포레스트(43.9점)와 선형 회귀(44.0점)의 차이는 0.07점이고, R²는 선형 회귀가 가장 높습니다. "
        "계수를 읽을 수 있고 수업에서 배운 선형 회귀를 최종 모델로 골랐습니다. "
        "XGBoost는 훈련 MAE(4.4점)와 시험 MAE(50.7점)의 차이가 커서 훈련 데이터를 외운 것에 가깝습니다."
    )

    st.subheader("변수 묶음 세 가지 비교 (선형 회귀, 시험 데이터)")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("체질량지수와 중성지방 두 개만 넣은 모델이 열 개를 모두 넣은 모델과 R²가 거의 같습니다.")

    which = st.radio("산점도로 볼 모델", list(SETS), index=1, horizontal=True)
    fig3 = px.scatter(result, x="progression", y=which, opacity=0.7,
                      labels={"progression": "실제 진행 점수", which: "예측 진행 점수"})
    fig3.add_trace(go.Scatter(x=[0, 350], y=[0, 350], mode="lines", name="예측 = 실제",
                              line=dict(dash="dash", color="gray")))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("점선 아래에 있는 점은 실제보다 낮게 예측한 환자입니다. 실제 점수가 높은 오른쪽 환자들이 대부분 점선 아래에 있습니다.")

    st.subheader("예측해 보기 (bmi + ltg 모델)")
    c1, c2 = st.columns(2)
    b = c1.slider("체질량지수(bmi)", float(df["bmi"].min()), float(df["bmi"].max()), 26.0, 0.1)
    l = c2.slider("중성지방 로그값(ltg)", float(df["ltg"].min()), float(df["ltg"].max()), 4.6, 0.01)
    pred = models["bmi + ltg"].predict(pd.DataFrame({"bmi": [b], "ltg": [l]}))[0]
    st.metric("예측 진행 점수", f"{pred:.1f}점")
    st.caption(f"ltg {l:.2f}는 중성지방 약 {np.exp(l):.0f}에 해당합니다(자연로그로 되돌린 값).")
    st.warning("수업용 분석입니다. 실제 환자의 병 진행을 예측하거나 치료를 정하는 데 사용하면 안 됩니다.")

with tab5:
    st.subheader("해석")
    st.markdown(
        "- 체질량지수 하나만 넣어도 평균 모델보다 MAE가 64.0점에서 52.3점으로 줄었습니다.\n"
        "- 중성지방(ltg)을 더하자 R²가 0.233에서 0.452로 올랐습니다. 측정값 열 개를 모두 넣어도 R²는 0.453으로 거의 같습니다.\n"
        "- 그래서 우리 팀의 답은 '예측할 수는 있지만 일부만이다. 그 일부는 대부분 체질량지수와 중성지방 두 수치로 설명된다'입니다.\n"
        "- 계수의 크기(ltg 67.1, bmi 5.8)는 단위가 달라 그대로 비교할 수 없습니다. 중요도는 상관계수와 변수를 넣고 빼는 실험으로 판단했습니다."
    )
    st.subheader("한계")
    st.markdown(
        "- R² 0.453은 진행 점수 차이의 절반 이상을 설명하지 못한다는 뜻입니다.\n"
        "- 시험 데이터에서 실제 점수가 250 이상인 환자 10명은 모두 실제보다 낮게 예측되었고, 평균 72.4점 낮았습니다.\n"
        "- 442명, 한 연구 집단의 데이터입니다. 다른 나라, 다른 시기의 환자에게 그대로 적용할 수 없습니다.\n"
        "- 이 모델로 개인의 병 진행을 예측하거나 치료 결정을 내리면 안 됩니다."
    )
