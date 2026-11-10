import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import filters, measure, morphology
import tempfile
import base64
import io
import os
import re
import zipfile
import gc

# =========================================================
#  0. パスワード認証 (必要に応じて secrets.toml で設定)
# =========================================================
def check_password():
    if "APP_PASSWORD" not in st.secrets:
        # Secretsに未設定の場合は開発用としてそのままパス
        return True

    correct_password = st.secrets["APP_PASSWORD"]

    def password_entered():
        if st.session_state["entered_password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["entered_password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 認証が必要です")
        st.caption("関係者専用ツールです。パスワードを入力してください。")
        st.text_input("パスワード", type="password", on_change=password_entered, key="entered_password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 認証が必要です")
        st.text_input("パスワード", type="password", on_change=password_entered, key="entered_password")
        st.error("パスワードが正しくありません。")
        return False
    return True

if not check_password():
    st.stop()


# =========================================================
#  1. q-value 計算モジュール
# =========================================================
try:
    from q_value import calculate_q_value
except ImportError:
    def calculate_q_value(bin_mask: np.ndarray) -> float:
        if np.count_nonzero(bin_mask) == 0:
            return 0.0
        dummy_q = float(np.sum(bin_mask)) / (bin_mask.size + 1e-6)
        return round(dummy_q, 4)


# =========================================================
#  2. .dat ファイル読み込みパーサー (x, y, value)
# =========================================================
def parse_dat_file(file_bytes: bytes) -> np.ndarray:
    """
    3列 (x, y, value) のテキストデータを2D numpy配列 (float32) に変換
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.strip().splitlines()
    
    xs, ys, vals = [], [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                xs.append(int(float(parts[0])))
                ys.append(int(float(parts[1])))
                vals.append(float(parts[2]))
            except ValueError:
                continue

    if not xs:
        raise ValueError("有効な数値データが見つかりませんでした。")

    max_x, max_y = max(xs), max(ys)
    W, H = max_x + 1, max_y + 1

    grid = np.zeros((H, W), dtype=np.float32)
    grid[ys, xs] = vals
    return grid


# =========================================================
#  3. 二値化処理パイプライン (提示コードと同一ロジック)
# =========================================================
def binarize_field_image(img_2d: np.ndarray, 
                         gauss_small=1.5, gauss_large=5.0, 
                         th_mode="手動調整", manual_th=0.10, 
                         min_area=11, max_area=100000, 
                         fill_holes=False, clip_percentile=99.7):
    img_float = img_2d.astype(np.float32)

    # Difference of Gaussians (DoG)
    blur_small = gaussian_filter(img_float, sigma=gauss_small)
    blur_large = gaussian_filter(img_float, sigma=gauss_large)
    dog_img = blur_small - blur_large
    dog_img[dog_img < 0] = 0.0

    # パーセンタイルクリッピングによる正規化
    pos_vals = dog_img[dog_img > 0]
    if len(pos_vals) > 0:
        v_high = float(np.percentile(pos_vals, clip_percentile))
    else:
        v_high = float(dog_img.max())

    if v_high > 1e-7:
        dog_clipped = np.clip(dog_img, 0, v_high)
        dog_norm = dog_clipped / v_high
    else:
        dog_norm = np.zeros_like(dog_img)

    # 閾値処理
    if th_mode == "Otsu（自動）":
        th_val = float(filters.threshold_otsu(dog_norm))
    else:
        th_val = float(manual_th)
    bw = dog_norm > th_val

    # 穴埋め
    if fill_holes:
        bw = binary_fill_holes(bw)

    # 粒子面積フィルタリング
    labeled = measure.label(bw)
    props = measure.regionprops(labeled)
    bw_filtered = np.zeros_like(bw, dtype=bool)
    for p in props:
        if min_area <= p.area <= max_area:
            bw_filtered[labeled == p.label] = True

    return bw_filtered, dog_norm, th_val


# =========================================================
#  4. PDF レポート生成
# =========================================================
def generate_dat_pdf_report(records, img_dict):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for idx, rec in enumerate(records):
            fname = rec["Filename"]
            imgs = img_dict[fname]

            raw_im = imgs["raw"]
            norm_dog = imgs["dog"]
            bw_im = imgs["mask"]

            # 正規化表示用
            r_disp = (raw_im - raw_im.min()) / (raw_im.max() - raw_im.min() + 1e-7)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            title_text = f"[{idx+1}] File: {fname} | Area Frac: {rec['Area_Fraction(%)']}% | q-value: {rec['q-value']} | Grid: {rec['Grid_Size']}"
            fig.suptitle(title_text, fontsize=10, y=1.02)

            axes[0].imshow(r_disp, cmap="viridis")
            axes[0].set_title("1. Raw Field (dat)", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(norm_dog, cmap="magma")
            axes[1].set_title("2. DoG Normalized", fontsize=10)
            axes[1].axis("off")

            axes[2].imshow(bw_im, cmap="gray")
            axes[2].set_title("3. Binary Mask", fontsize=10)
            axes[2].axis("off")

            # オーバーレイ (背景グレー + マスク赤)
            overlay = np.stack([r_disp]*3, axis=-1)
            overlay[bw_im] = [1.0, 0.2, 0.2]
            axes[3].imshow(np.clip(overlay, 0.0, 1.0))
            axes[3].set_title("4. Overlay", fontsize=10)
            axes[3].axis("off")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf_buffer.seek(0)
    data = pdf_buffer.getvalue()
    pdf_buffer.close()
    return data


# =========================================================
#  5. Streamlit UI
# =========================================================
st.set_page_config(page_title=".dat 二値化・解析システム", layout="wide")

st.title("🧪 .dat グリッドデータ 一括二値化・q値解析システム")
st.caption("x y value 形式の .dat データを読み込み、DoGフィルタ＋閾値処理で二値化画像を自動生成します。")

# サイドバー：パラメータ設定
st.sidebar.header("🛠 二値化パラメータ設定")
p_small = st.sidebar.slider("DoG 小シグマ (Gauss Small)", 0.5, 5.0, 1.5, step=0.1)
p_large = st.sidebar.slider("DoG 大シグマ (Gauss Large)", 2.0, 15.0, 5.0, step=0.5)
p_th_mode = st.sidebar.radio("閾値判定方式", ["Otsu（自動）", "手動調整"], index=1)
p_manual_th = st.sidebar.slider("手動閾値 (Threshold)", 0.0, 1.0, 0.10, step=0.01)
p_clip = st.sidebar.slider("上限クリッピング (%)", 90.0, 100.0, 99.7, step=0.1)
p_min_area = st.sidebar.number_input("最小粒子面積 (px)", min_value=1, value=11, step=1)
p_max_area = st.sidebar.number_input("最大粒子面積 (px)", min_value=10, max_value=1000000, value=100000, step=1000)
p_fill = st.sidebar.checkbox("マスク内の穴を埋める", value=False)

# ファイルアップロード
uploaded_files = st.file_uploader(
    "解析対象の .dat ファイルをまとめてアップロードしてください（複数可）",
    type=["dat", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    records = []
    zip_images = []
    cache_images = {}

    st.subheader(f"📁 アップロード済み: {len(uploaded_files)} ファイル")

    # プログレスバー
    progress_bar = st.progress(0.0)
    
    for idx, f in enumerate(uploaded_files):
        base_name, _ = os.path.splitext(f.name)
        f.seek(0)
        bytes_data = f.read()

        try:
            raw_grid = parse_dat_file(bytes_data)
        except Exception as e:
            st.error(f"ファイル解析エラー ({f.name}): {e}")
            continue

        H, W = raw_grid.shape
        total_pixels = H * W

        # 二値化実行
        bw_mask, dog_norm, used_th = binarize_field_image(
            raw_grid,
            gauss_small=p_small,
            gauss_large=p_large,
            th_mode=p_th_mode,
            manual_th=p_manual_th,
            min_area=p_min_area,
            max_area=p_max_area,
            fill_holes=p_fill,
            clip_percentile=p_clip
        )

        mask_pixels = int(np.count_nonzero(bw_mask))
        area_frac = round((mask_pixels / total_pixels) * 100, 3) if total_pixels > 0 else 0.0
        q_val = calculate_q_value(bw_mask)

        # レコード蓄積
        records.append({
            "Filename": f.name,
            "Grid_Size": f"{W}x{H}",
            "Total_Pixels": total_pixels,
            "Positive_Pixels": mask_pixels,
            "Area_Fraction(%)": area_frac,
            "q-value": q_val,
            "Applied_Threshold": round(used_th, 4),
            "Min_Value": round(float(raw_grid.min()), 4),
            "Max_Value": round(float(raw_grid.max()), 4),
            "Mean_Value": round(float(raw_grid.mean()), 4)
        })

        # PNG バイナリ生成
        buf_mask = io.BytesIO()
        Image.fromarray((bw_mask.astype(np.uint8) * 255)).save(buf_mask, format="PNG")
        mask_bytes = buf_mask.getvalue()

        zip_images.append({
            "filename": f"{base_name}_binary_mask.png",
            "data": mask_bytes
        })

        cache_images[f.name] = {
            "raw": raw_grid,
            "dog": dog_norm,
            "mask": bw_mask,
            "mask_bytes": mask_bytes
        }

        progress_bar.progress((idx + 1) / len(uploaded_files))

    # --- プレビュー表示 ---
    st.markdown("---")
    st.subheader("👁️ プレビュー表示")
    file_options = [r["Filename"] for r in records]
    selected_view = st.selectbox("プレビューするファイルを選択", file_options)

    if selected_view:
        sel_rec = next(r for r in records if r["Filename"] == selected_view)
        sel_imgs = cache_images[selected_view]

        p_col1, p_col2, p_col3 = st.columns(3)
        raw_disp = (sel_imgs["raw"] - sel_imgs["raw"].min()) / (sel_imgs["raw"].max() - sel_imgs["raw"].min() + 1e-7)

        p_col1.image(raw_disp, caption=f"元データ (Normalized {sel_rec['Grid_Size']})", width="stretch")
        p_col2.image(sel_imgs["dog"], caption=f"DoG フィルタ後", width="stretch")
        p_col3.image(sel_imgs["mask"].astype(np.uint8) * 255, caption=f"二値化マスク (面積分率: {sel_rec['Area_Fraction(%)']}%)", width="stretch")

        st.download_button(
            label="📥 選択中画像の二値化PNGを保存",
            data=sel_imgs["mask_bytes"],
            file_name=f"{os.path.splitext(selected_view)[0]}_binary.png",
            mime="image/png"
        )

    # --- 累積解析テーブル ＆ ダウンロード ---
    st.markdown("---")
    st.subheader("📋 一括解析結果テーブル")
    df_results = pd.DataFrame(records)
    st.dataframe(df_results, use_container_width=True)

    # CSV
    csv_bytes = df_results.to_csv(index=False).encode("utf-8_sig")

    # ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in zip_images:
            zf.writestr(item["filename"], item["data"])
    zip_bytes = zip_buf.getvalue()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="📊 解析CSVをダウンロード",
            data=csv_bytes,
            file_name="dat_binarization_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    with c2:
        st.download_button(
            label=f"📦 二値化マスク画像一括ZIP保存 ({len(zip_images)} 枚)",
            data=zip_bytes,
            file_name="dat_binary_masks.zip",
            mime="application/zip",
            use_container_width=True
        )
    with c3:
        if st.button("📄 全ファイルの画像PDFレポートを生成", use_container_width=True):
            with st.spinner("PDFレポートを生成中..."):
                pdf_data = generate_dat_pdf_report(records, cache_images)
                st.download_button(
                    label=f"⬇️ クリックしてPDFを保存 ({len(records)} ページ)",
                    data=pdf_data,
                    file_name="dat_analysis_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
else:
    st.info("上のエリアに複数の .dat ファイルをドラッグ＆ドロップすると、自動で全件解析が実行されます。")