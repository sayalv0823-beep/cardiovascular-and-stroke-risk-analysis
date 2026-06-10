# ライブラリの読み込み
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import streamlit as st
import plotly.express as px

# データセットの読み込み
df_heart_jp_after = pd.read_csv("df_heart_jp_after.csv")
df_stroke_model = pd.read_csv("df_stroke_model.csv")

# BMIの判定関数
def judge_bmi(bmi):
    if bmi < 18.5:
        return "低体重"
    elif bmi < 25:
        return "普通体重"
    elif bmi < 30:
        return "肥満（1度）"
    elif bmi < 35:
        return "肥満（2度）"
    elif bmi < 40:
        return "肥満（3度）"
    else:
        return "肥満（4度）"

# 収縮期血圧の判定関数
def judge_bp(systolic_bp):
    if systolic_bp >= 140:
        return "高血圧判定あり"
    elif systolic_bp >= 130:
        return "やや高め"
    else:
        return "基準範囲内"
    
# 年齢に近い人の平均血糖値を取得する関数

def get_average_glucose_by_age(age):
    age_min = age - 5
    age_max = age + 5

    same_age_df = df_heart_jp_after[
        (df_heart_jp_after["年齢"] >= age_min) &
        (df_heart_jp_after["年齢"] <= age_max)
    ]

    return same_age_df["血糖値"].mean()

# 心疾患最終モデル

features_heart = [
    "性別",
    "年齢",
    "1日喫煙本数",
    "脳卒中既往",
    "高血圧既往",
    "収縮期血圧",
    "血糖値"
]

target_heart = "10年後心疾患リスク"

X_heart = df_heart_jp_after[features_heart]
y_heart = df_heart_jp_after[target_heart]

X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(
    X_heart,
    y_heart,
    test_size=0.2,
    random_state=42,
    stratify=y_heart
)

X_heart_train_const = sm.add_constant(X_heart_train)
X_heart_test_const = sm.add_constant(X_heart_test)

heart_model = sm.Logit(y_heart_train, X_heart_train_const)
heart_result = heart_model.fit()

y_heart_pred_prob = heart_result.predict(X_heart_test_const)
y_heart_pred = (y_heart_pred_prob >= 0.5).astype(int)

heart_accuracy = accuracy_score(y_heart_test, y_heart_pred)
heart_auc = roc_auc_score(y_heart_test, y_heart_pred_prob)

print("心疾患 Accuracy:", heart_accuracy)
print("心疾患 AUC:", heart_auc)


# 脳卒中最終モデル

features_stroke_simple = [
    "年齢",
    "高血圧",
    "心疾患",
    "平均血糖値",
    "BMI",
    "性別_男性",
    "喫煙状況_喫煙中",
    "喫煙状況_過去喫煙",
    "喫煙状況_非喫煙"
]

target_stroke = "脳卒中"

X_stroke_simple = df_stroke_model[features_stroke_simple]
y_stroke_simple = df_stroke_model[target_stroke]

X_stroke_simple = X_stroke_simple.astype(float)

X_stroke_train, X_stroke_test, y_stroke_train, y_stroke_test = train_test_split(
    X_stroke_simple,
    y_stroke_simple,
    test_size=0.2,
    random_state=42,
    stratify=y_stroke_simple
)

# 定数項追加
X_stroke_train_const = sm.add_constant(
    X_stroke_train,
    has_constant="add"
)

X_stroke_test_const = sm.add_constant(
    X_stroke_test,
    has_constant="add"
)

# ロジスティック回帰
stroke_simple_model = sm.Logit(
    y_stroke_train,
    X_stroke_train_const
)

stroke_simple_result = stroke_simple_model.fit()

print(stroke_simple_result.summary())

# 予測
y_stroke_pred_prob = stroke_simple_result.predict(X_stroke_test_const)
y_stroke_pred = (y_stroke_pred_prob >= 0.5).astype(int)

# 評価
stroke_simple_accuracy = accuracy_score(y_stroke_test, y_stroke_pred)
stroke_simple_auc = roc_auc_score(y_stroke_test, y_stroke_pred_prob)

print("脳卒中簡易モデル Accuracy:", stroke_simple_accuracy)
print("脳卒中簡易モデル AUC:", stroke_simple_auc)

# 心疾患予測関数
def predict_heart_risk_app(
    性別,
    年齢,
    喫煙状況,
    喫煙本数=0,
    脳卒中既往="なし",
    高血圧既往="なし",
    収縮期血圧=120,
    血糖値=100
):

    性別_num = 1 if 性別 == "男性" else 0

    喫煙本数_num = 0 if 喫煙状況 == "非喫煙" else 喫煙本数

    脳卒中既往_num = 1 if 脳卒中既往 == "あり" else 0
    高血圧既往_num = 1 if 高血圧既往 == "あり" else 0

    input_data = pd.DataFrame({
        "性別": [性別_num],
        "年齢": [年齢],
        "1日喫煙本数": [喫煙本数_num],
        "脳卒中既往": [脳卒中既往_num],
        "高血圧既往": [高血圧既往_num],
        "収縮期血圧": [収縮期血圧],
        "血糖値": [血糖値]
    })

    input_data_const = sm.add_constant(
        input_data,
        has_constant="add"
    )

    risk = heart_result.predict(input_data_const)[0]

    return risk

# 脳卒中リスク予測関数（選択肢入力版・標準化なし）

def predict_stroke_risk_app(
    年齢,
    高血圧,
    心疾患,
    平均血糖値,
    BMI,
    性別,
    喫煙状況
):

    性別_男性 = 1 if 性別 == "男性" else 0

    喫煙状況_喫煙中 = 1 if 喫煙状況 == "喫煙中" else 0
    喫煙状況_過去喫煙 = 1 if 喫煙状況 == "過去喫煙" else 0
    喫煙状況_非喫煙 = 1 if 喫煙状況 == "非喫煙" else 0

    input_data = pd.DataFrame({
        "年齢": [年齢],
        "高血圧": [高血圧],
        "心疾患": [心疾患],
        "平均血糖値": [平均血糖値],
        "BMI": [BMI],
        "性別_男性": [性別_男性],
        "喫煙状況_喫煙中": [喫煙状況_喫煙中],
        "喫煙状況_過去喫煙": [喫煙状況_過去喫煙],
        "喫煙状況_非喫煙": [喫煙状況_非喫煙]
    })

    input_data_const = sm.add_constant(
        input_data,
        has_constant="add"
    )

    risk = stroke_simple_result.predict(input_data_const)[0]

    return risk

# Streamlitアプリの画面構築



st.set_page_config(
    page_title="健康リスク予測アプリ",
    page_icon="🩷",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFDFC;
    }

    .warm-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-left: 8px solid #C9788D;
    }

    .note-box {
        background-color: #FAF3F4;
        padding: 18px;
        border-radius: 14px;
        border-left: 4px solid #C9788D;
        margin-bottom: 20px;
        color: #5f4b4b;
    }

    .soft-text {
        color: #5f4b4b;
        font-size: 16px;
        line-height: 1.7;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)   

st.markdown(
    """
    <div class="warm-card">
        <h1 style="font-size: 36px; line-height: 1.3;">
            🩷 心疾患・脳卒中リスクチェック
        </h1>
        <p class="soft-text">
        年齢、血圧、喫煙状況などから、心疾患と脳卒中のリスクを予測します。<br>
        生活習慣を見直すきっかけとして使える、参考用のアプリです。
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note-box">
        <strong>このアプリについて</strong><br>
        このアプリは医療診断を行うものではありません。<br>
        使用しているデータは日本人専用のデータではないため、表示されるリスクはあくまで参考値です。<br>
        気になる症状や健康不安がある場合は、医療機関に相談してください。
    </div>
    """,
    unsafe_allow_html=True
)

st.write("入力情報から、心疾患リスクと脳卒中リスクを予測します。")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("基本情報")

    性別 = st.selectbox(
        "性別",
        ["選択してください", "女性", "男性"]
    )

    年齢_input = st.text_input(
        "年齢",
        placeholder="例：40"
    )

    身長_input = st.text_input(
        "身長(cm)",
        placeholder="例：160"
    )

    体重_input = st.text_input(
        "体重(kg)",
        placeholder="例：55"
    )

with right_col:
    st.subheader("生活習慣・既往歴")

    喫煙状況 = st.selectbox(
        "喫煙状況",
        ["選択してください", "非喫煙", "喫煙中", "過去喫煙"]
    )

    if 喫煙状況 == "喫煙中":
        喫煙本数_input = st.text_input(
            "1日喫煙本数",
            placeholder="例：10"
        )
    else:
        喫煙本数_input = "0"

    脳卒中既往 = st.selectbox(
        "脳卒中になったことがありますか？",
        ["選択してください", "なし", "あり"]
    )

    心疾患既往 = st.selectbox(
        "心疾患になったことがありますか？",
        ["選択してください", "なし", "あり"]
    )

    高血圧既往 = st.selectbox(
        "高血圧と言われたことがありますか？",
        ["選択してください", "なし", "あり"]
    )

    収縮期血圧_input = st.text_input(
        "上の血圧",
        placeholder="例：120"
    )

    血糖値不明 = st.checkbox(
        "血糖値が分からない"
    )

    if 血糖値不明:
        血糖値_input = ""
        st.write("血糖値は同年代の平均値を使って計算します。")
    else:
        血糖値_input = st.text_input(
            "血糖値",
            placeholder="例：100"
        )
st.divider()

if "result_calculated" not in st.session_state:
    st.session_state.result_calculated = False

if st.button("リスクを計算する", use_container_width=True):
    required_select_items = [
        性別,
        喫煙状況,
        脳卒中既往,
        心疾患既往,
        高血圧既往
    ]

    if "選択してください" in required_select_items:
        st.warning("未選択の項目があります。すべて選択してください。")
        st.stop()

    if 年齢_input == "":
        st.warning("年齢を入力してください。")
        st.stop()

    if 身長_input == "":
        st.warning("身長を入力してください。")
        st.stop()

    if 体重_input == "":
        st.warning("体重を入力してください。")
        st.stop()

    if 収縮期血圧_input == "":
        st.warning("上の血圧を入力してください。")
        st.stop()

    if 喫煙状況 == "喫煙中" and 喫煙本数_input == "":
        st.warning("喫煙中の場合は、1日喫煙本数を入力してください。")
        st.stop()

    if (not 血糖値不明) and 血糖値_input == "":
        st.warning("血糖値を入力するか、「血糖値が分からない」にチェックしてください。")
        st.stop()

    年齢 = int(年齢_input)
    身長 = float(身長_input)
    体重 = float(体重_input)
    収縮期血圧 = int(収縮期血圧_input)
    喫煙本数 = int(喫煙本数_input)

    BMI = 体重 / ((身長 / 100) ** 2)
    BMI判定 = judge_bmi(BMI)

    if 血糖値不明:
        血糖値 = get_average_glucose_by_age(年齢)
        血糖値メモ = f"血糖値は同年代平均の {血糖値:.1f} を使用"
    else:
        血糖値 = float(血糖値_input)
        血糖値メモ = f"入力された血糖値 {血糖値:.1f} を使用"

    平均血糖値 = 血糖値

    st.divider()

    血圧判定 = judge_bp(収縮期血圧)

    高血圧_num = 1 if (高血圧既往 == "あり" or 収縮期血圧 >= 140) else 0
    心疾患_num = 1 if 心疾患既往 == "あり" else 0

    st.session_state.BMI = BMI
    st.session_state.BMI判定 = BMI判定
    st.session_state.血圧判定 = 血圧判定
    st.session_state.血糖値メモ = 血糖値メモ

    required_items = [
        性別,
        喫煙状況,
        脳卒中既往,
        心疾患既往,
        高血圧既往
    ]

    if "選択してください" in required_items:
        st.warning("未入力の項目があります。すべて選択してから計算してください。")
        st.stop()

    heart_risk = predict_heart_risk_app(
        性別=性別,
        年齢=年齢,
        喫煙状況=喫煙状況,
        喫煙本数=喫煙本数,
        脳卒中既往=脳卒中既往,
        高血圧既往=高血圧既往,
        収縮期血圧=収縮期血圧,
        血糖値=血糖値
    )

    stroke_risk = predict_stroke_risk_app(
        年齢=年齢,
        高血圧=高血圧_num,
        心疾患=心疾患_num,
        平均血糖値=平均血糖値,
        BMI=BMI,
        性別=性別,
        喫煙状況=喫煙状況
    )

    age_min = 年齢 - 5
    age_max = 年齢 + 5

    same_age_heart_risk = df_heart_jp_after[
        (df_heart_jp_after["年齢"] >= age_min) &
        (df_heart_jp_after["年齢"] <= age_max)
    ]["10年後心疾患リスク"].mean()

    same_age_stroke_risk = df_stroke_model[
        (df_stroke_model["年齢"] >= age_min) &
        (df_stroke_model["年齢"] <= age_max)
    ]["脳卒中"].mean()

    st.session_state.result_calculated = True
    st.session_state.heart_risk = heart_risk
    st.session_state.stroke_risk = stroke_risk
    st.session_state.same_age_heart_risk = same_age_heart_risk
    st.session_state.same_age_stroke_risk = same_age_stroke_risk

if st.session_state.result_calculated:

    heart_risk = st.session_state.heart_risk
    stroke_risk = st.session_state.stroke_risk
    same_age_heart_risk = st.session_state.same_age_heart_risk
    same_age_stroke_risk = st.session_state.same_age_stroke_risk

    heart_diff = heart_risk - same_age_heart_risk
    stroke_diff = stroke_risk - same_age_stroke_risk

    def compare_message(diff):
        if abs(diff) < 0.005:
            return "同年代平均と同程度です"
        elif diff < 0:
            return "同年代平均より低めです"
        else:
            return "同年代平均より高めです"

    heart_message = compare_message(heart_diff)
    stroke_message = compare_message(stroke_diff)

    st.header("予測結果")
    st.markdown("### 入力値の確認")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.info(f"BMI：{st.session_state.BMI:.1f}（{st.session_state.BMI判定}）")

    with info_col2:
        st.info(f"血圧判定：{st.session_state.血圧判定}")

    with info_col3:
        st.info(st.session_state.血糖値メモ)

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown(
            f"""
            <div style="
                background-color:#fff7f9;
                padding:24px;
                border-radius:20px;
                box-shadow:0 4px 12px rgba(0,0,0,0.08);
                border-left:8px solid #C9788D;
            ">
                <h3>💗 心疾患リスク</h3>
                <div style="font-size:42px; font-weight:700; color:#C9788D;">
                    {heart_risk:.1%}
                </div>
                <p>{heart_message}</p>
                <p style="color:#777;">同年代平均との差：{heart_diff:+.1%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col2:
        st.markdown(
            f"""
            <div style="
                background-color:#fff7f9;
                padding:24px;
                border-radius:20px;
                box-shadow:0 4px 12px rgba(0,0,0,0.08);
                border-left:8px solid #D98FA0;
            ">
                <h3>🧠 脳卒中リスク</h3>
                <div style="font-size:42px; font-weight:700; color:#C9788D;">
                    {stroke_risk:.1%}
                </div>
                <p>{stroke_message}</p>
                <p style="color:#777;">同年代平均との差：{stroke_diff:+.1%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        heart_compare_df = pd.DataFrame({
            "区分": ["あなた", "同年代平均"],
            "リスク(%)": [heart_risk * 100, same_age_heart_risk * 100]
        })

        fig_heart = px.bar(
            heart_compare_df,
            x="区分",
            y="リスク(%)",
            text="リスク(%)",
            color="区分",
            color_discrete_map={
                "あなた": "#D98FA0",
                "同年代平均": "#F3D8DE"
            }
        )

        fig_heart.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="inside"
        )

        fig_heart.update_layout(
            title="心疾患リスク",
            showlegend=False,
            height=320,
            yaxis_title="リスク（%）",
            xaxis_title=None,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_heart, use_container_width=True)

    with graph_col2:
        stroke_compare_df = pd.DataFrame({
            "区分": ["あなた", "同年代平均"],
            "リスク(%)": [stroke_risk * 100, same_age_stroke_risk * 100]
        })

        fig_stroke = px.bar(
            stroke_compare_df,
            x="区分",
            y="リスク(%)",
            text="リスク(%)",
            color="区分",
            color_discrete_map={
                "あなた": "#D98FA0",
                "同年代平均": "#F3D8DE"
            }
        )

        fig_stroke.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="inside"
        )

        fig_stroke.update_layout(
            title="脳卒中リスク",
            showlegend=False,
            height=320,
            yaxis_title="リスク（%）",
            xaxis_title=None,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_stroke,
            use_container_width=True
        )