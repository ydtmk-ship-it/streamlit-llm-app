# app.py ーー ルームツアー台本ジェネレーター（画像/PDF対応・LangChain）

import os, io, base64
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF（PDF→画像化）
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =============================
# 初期化：キー読み込み（ローカル=.env / クラウド=Secrets）
# =============================
from dotenv import load_dotenv
import os, streamlit as st

load_dotenv()  # ローカルは .env を読む

# ローカル(.env)が無いときだけ、secrets を試す（無ければ黙ってスルー）
if not os.getenv("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

# =============================
# 画像/ PDF ヘルパー
# =============================
def _img_bytes_to_data_url(data: bytes) -> str:
    """画像バイト列をJPEGにして data URL へ"""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _pdf_bytes_to_data_urls(pdf_bytes: bytes, max_pages: int = 2) -> List[str]:
    """PDFの先頭から max_pages ページを画像化して data URL リストを返す"""
    urls: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            # 2x 拡大で少し高解像度に
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


# =============================
# Brief（プロンプト本文）整形
# =============================
def make_brief(
    title: str,
    area: str,
    floor_tsubo: str,
    family: str,
    target: str,
    perf: List[str],
    highlights: List[str],
    focus: List[str],
    tone: str,
    minutes: int,
    budget: str,
    notes: str,
) -> str:
    def join(items: List[str]) -> str:
        return "、".join([x for x in items if x]) if items else "（未指定）"

    lines = [
        f"案件名: {title or '（未指定）'}",
        f"エリア: {area or '（未指定）'}",
        f"延床: {floor_tsubo or '（未指定）'}坪",
        f"家族構成: {family or '（未指定）'}",
        f"ターゲット: {target or '（未指定）'}",
        f"性能・設備: {join(perf)}",
        f"見どころ: {join(highlights)}",
        f"強調したい価値: {join(focus)}",
        f"トーン: {tone}",
        f"想定尺: {minutes}分",
    ]
    if budget:
        lines.append(f"予算帯: {budget}")
    if notes:
        lines.append(f"補足: {notes}")

    lines.append("— 以上の情報と添付の間取り図（画像/PDF）を踏まえて、出力フォーマットに沿って台本を作成してください。")
    return "\n".join(lines)


# =============================
# 台本生成（LLM呼び出し：この課題の必須関数）
#   引数：brief（入力まとめ）、persona（ラジオ選択値）、uploads（画像/PDF）
#   戻り値：台本テキスト
# =============================
def generate_script(brief: str, persona: str, uploads: Optional[List] = None) -> str:
    system_map = {
        "設計士（性能・間取り重視）": (
            "あなたは住宅の設計士。耐震・断熱・動線・収納・空調などの性能面を中心に、"
            "専門用語をかみ砕き、視聴者が納得する根拠とベネフィットを端的に説明してください。"
        ),
        "インテリアコーディネーター（デザイン重視）": (
            "あなたはインテリアコーディネーター。素材・色・照明・造作の意図を言語化し、"
            "写真映えするカット指示や小物提案も添えて解説してください。"
        ),
        "営業・MC（親しみやすくカジュアル）": (
            "あなたは住宅会社のMC。親しみやすい語り口で生活シーンを描写し、"
            "わかりやすいベネフィットと軽快なテンポで案内してください。"
        ),
    }
    system_msg = (
        system_map.get(persona, "あなたは有能なアシスタントです。")
        + " 出力は日本語。指定がなければ15分尺を想定し、各セクションに目安時間を入れてください。\n"
        + "\n【出力フォーマット】\n"
        + "1) オープニング（0:00-）: 視聴者へのフック / 家の特徴3点 / 本日の流れ\n"
        + "2) 外観: 立地・外観デザイン・駐車・外構（B-rollとパンの指示）\n"
        + "3) 玄関〜土間: 動線・収納（ショット/カメラ動き/一言テロップ案）\n"
        + "4) 1F: LDK / キッチン / 水回り（理由→ベネフィットで解説）\n"
        + "5) 2F: 個室・主寝室・書斎・家事動線\n"
        + "6) 性能・コスト: 断熱/耐震/空調/省エネと概算コストの目安\n"
        + "7) エンディング: まとめ / CTA（チャンネル登録・見学予約）\n"
        + "※ 各セクションに『撮影カット指示（ショット/カメラ動き/B-roll）』と『一言テロップ案』を付与。"
    )

    if not os.getenv("OPENAI_API_KEY"):
        return "環境変数 OPENAI_API_KEY が見つかりません。.env もしくは Secrets を確認してください。"

    # 画像/PDF（任意）をメッセージに添付（OpenAIのマルチモーダル形式）
    contents = []

    if uploads:
        try:
            for f in uploads:
                # UploadedFile → バイト列
                try:
                    data = f.getvalue()
                except Exception:
                    f.seek(0)
                    data = f.read()

                name = (getattr(f, "name", "") or "").lower()
                mime = (getattr(f, "type", "") or "").lower()

                if name.endswith(".pdf") or mime == "application/pdf":
                    for url in _pdf_bytes_to_data_urls(data, max_pages=2):
                        contents.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    url = _img_bytes_to_data_url(data)
                    contents.append({"type": "image_url", "image_url": {"url": url}})
        except Exception as e:
            contents.append({"type": "text", "text": f"[警告] 添付の読み込みに失敗: {e}"})

    contents.append({"type": "text", "text": brief})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, timeout=90)
    result = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=contents)])
    return result.content


# =============================
# UI
# =============================
st.set_page_config(page_title="ルームツアー台本ジェネレーター", page_icon="🏠")
st.title("🏠 ルームツアー台本ジェネレーター（画像/PDFの間取り図対応）")
with st.expander("ℹ️ アプリの概要と使い方（必読）", expanded=True):
    st.markdown(
        """
1. **スタイル**（ラジオ）を選ぶ  
2. **各項目**を入力（所在地/延床/家族構成/見どころ/トーン/尺など）  
3. **間取り図**をアップロード（**PNG/JPG/WebP または PDF**、複数可）  
4. **台本を生成**を押す  

- ローカルでは `.env` の `OPENAI_API_KEY` を読み込み、クラウドでは **Secrets** を使います。  
- PDFは**先頭から最大2ページ**を自動で画像化して解析に使います。
        """
    )

persona = st.radio(
    "ステップ1：ナレーションのスタイルを選択",
    [
        "設計士（性能・間取り重視）",
        "インテリアコーディネーター（デザイン重視）",
        "営業・MC（親しみやすくカジュアル）",
    ],
)

uploads = st.file_uploader(
    "ステップ2：間取り図（画像またはPDF／任意・複数可）",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
    accept_multiple_files=True,
)

# 簡易プレビュー
if uploads:
    img_files = [f for f in uploads if (f.type or "").startswith("image")]
    pdf_files = [f for f in uploads if (f.type or "") == "application/pdf" or f.name.lower().endswith(".pdf")]

    if img_files:
        st.caption(f"画像 {len(img_files)} 件を読み込みました。プレビュー：")
        st.image(img_files, caption=[f.name for f in img_files], width=240)

    if pdf_files:
        st.caption("PDFを読み込みました： " + ", ".join(f.name for f in pdf_files))
        st.caption("※ PDFは先頭ページ等を内部で画像化してLLMに渡します。")

with st.form("script_form"):
    st.subheader("ステップ3：物件情報の入力")
    title = st.text_input("案件名/物件名", placeholder="例：モデルハウス 伊丹市船原")
    c1, c2 = st.columns(2)
    with c1:
        area = st.text_input("所在地/エリア", placeholder="例：札幌市豊平区")
        family = st.text_input("家族構成", placeholder="例：夫婦+子ども2人")
        budget = st.text_input("予算帯（任意）", placeholder="例：本体価格3,000万円前後")
    with c2:
        floor_tsubo = st.text_input("延床面積（坪）", placeholder="例：34")
        target = st.selectbox("ターゲット", ["子育て世帯", "共働き", "二世帯", "DINKs/単身", "シニア", "その他"], index=0)
        tone = st.selectbox("トーン", ["親しみやすい", "落ち着いた", "専門的", "エモーショナル"], index=0)

    perf = st.multiselect(
        "性能・設備（複数選択可）",
        ["耐震等級3", "断熱等級6", "全館空調", "太陽光", "蓄電池", "V2H", "床暖房", "トリプルサッシ", "第一種換気"],
        default=["耐震等級3", "断熱等級6"],
    )
    highlights = st.multiselect(
        "見どころ（複数選択可）",
        [
            "アイアン手摺",
            "モールテックス",
            "造作洗面",
            "回遊動線",
            "吹き抜け",
            "ただいま動線",
            "ランドリールーム",
            "ファミクロ",
            "パントリー",
            "書斎",
            "ヌック",
            "土間収納",
        ],
        default=["アイアン手摺", "造作洗面"],
    )
    add_extra = st.text_input("その他の見どころ（カンマ区切りで追加）", "")
    focus = st.multiselect(
        "強調したい価値",
        ["家事動線", "収納計画", "光熱費の根拠", "メンテナンス性", "音/断熱/窓", "セキュリティ", "デザインコンセプト"],
        default=["家事動線", "収納計画"],
    )
    minutes = st.slider("想定尺（分）", 5, 50, 15, 5)

    notes = st.text_area("補足・要望（任意）", height=120)
    submitted = st.form_submit_button("台本を生成")

if submitted:
    extra_list = [s.strip() for s in (add_extra or "").split(",") if s.strip()]
    brief = make_brief(
        title=title,
        area=area,
        floor_tsubo=floor_tsubo,
        family=family,
        target=target,
        perf=perf,
        highlights=highlights + extra_list,
        focus=focus,
        tone=tone,
        minutes=minutes,
        budget=budget,
        notes=notes,
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が見つかりません。.env または Secrets を設定してください。")
    elif not any([title, area, floor_tsubo, family, notes, uploads]):
        st.warning("最低1つ以上の項目またはファイルを入力/アップロードしてください。")
    else:
        with st.spinner("台本を作成中…"):
            script = generate_script(brief, persona, uploads)
        st.markdown("### 🎬 生成された台本")
        st.write(script)

st.divider()
st.caption("© 2025 Room Tour Script Generator. For educational use.")