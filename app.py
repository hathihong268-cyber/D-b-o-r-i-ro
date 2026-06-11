import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1) SET PAGE CONFIG (Lệnh Streamlit đầu tiên)
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Gian Lận tại Agribank",
    page_icon="❤️"
)

# 2) IMPORT & CÁC HÀM CACHE DÙNG CHUNG
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hash (cache)"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(format(e))
        return None

# 3) SIDEBAR (TP1) - VÙNG CẤU HÌNH
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (CSV hoặc Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Chọn tệp dữ liệu mẫu chứa các cột từ X_1 đến X_14 và cột mục tiêu 'default'"
    )
    
    st.divider()
    st.subheader("🤖 Tham số mô hình AI")
    st.caption("Mô hình: RandomForestClassifier (Model3)")
    
    # Trích xuất siêu tham số từ cấu hình mặc định hợp lý dựa trên notebook
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đo lường (criterion)", 
        options=["gini", "entropy", "log_loss"], index=0,
        help="Chức năng đo lường chất lượng phân tách."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, max_value=50, value=15,
        help="Độ sâu tối đa của các cây quyết định."
    )
    
    random_state = st.number_input(
        "Mã ngẫu nhiên (random_state)", 
        value=42, step=1,
        help="Đảm bảo tính tái lập của kết quả huấn luyện."
    )
    
    # Khối nâng cao trong expander
    with st.expander("Tham số nâng cao"):
        min_samples_split = st.slider("Min samples split", min_value=2, max_value=10, value=2)
        test_size = st.slider("Tỷ lệ tập kiểm tra (Test size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.divider()
    # Nút bấm huấn luyện duy nhất
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# 4) HEADER (TP2) - VÙNG ĐỊNH HƯỚNG
st.title("🛡️ Hệ Thống Phát Hiện Gian Lận tại Agribank")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch, đánh giá hành vi rủi ro tài chính và phân loại tự động bằng học máy.")

if uploaded_file is None:
    st.info("👋 Chào mừng bạn! Vui lòng tải tệp dữ liệu ở thanh bên (Sidebar) để bắt đầu sử dụng ứng dụng.")
    st.stop(❤️)
else:
    # Đọc dữ liệu qua hàm cache
    file_bytes = uploaded_file.getvalue()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("❌ Không thể đọc tệp dữ liệu. Vui lòng kiểm tra lại định dạng.")
        st.stop()
        
    st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | Kích thước: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột")
st.divider()

# ĐỊNH NGHĨA DANH SÁCH BIẾN DỰA TRÊN NOTEBOOK & DATASET
features = [f"X_{i}" for i in range(1, 15)]
target = "default"

# 5) KHỐI TRAIN (Chạy khi bấm nút, lưu kết quả vào session_state)
if train_clicked:
    # Kiểm tra schema hợp lệ
    missing_cols = [col for col in features + [target] if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc: {missing_cols}")
    else:
        with st.spinner("⏳ Đang huấn luyện mô hình Random Forest..."):
            X = df_raw[features]
            y = df_raw[target]
            
            # Chia tập dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Khởi tạo và fit mô hình
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Dự đoán và tính toán kết quả kiểm định
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Lưu vào session_state 3 thành phần cốt lõi
            st.session_state['trained_model'] = model
            st.session_state['features_list'] = features
            st.session_state['results'] = {
                'y_test': y_test.values,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'cm': confusion_matrix(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
        st.success("🎉 Huấn luyện mô hình thành công! Kết quả đã được cập nhật ở các Tab dưới đây.")

# 6) KHỞI TẠO CÁC TABS ĐIỀU HƯỚNG CHÍNH
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Dự báo & Sử dụng"
])

# ---------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ---------------------------------------------------------
with tab1:
    st.subheader("📋 Phân tích thống kê mô tả")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số lượng dòng (Rows)", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng cột (Columns)", f"{df_raw.shape[1]:,}")
    col_m3.metric("Dung lượng File", f"{uploaded_file.size / (1024*1024):.2f} MB")
    
    st.write("##### 🔍 Xem trước 5 dòng dữ liệu đầu tiên (Head):")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    st.write("##### 📈 Bảng mô tả thống kê các biến mô hình (X & y):")
    cols_to_describe = [col for col in features + [target] if col in df_raw.columns]
    st.dataframe(df_raw[cols_to_describe].describe(), use_container_width=True)

# ---------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Biểu đồ phân phối các biến quan trọng")
    
    # Lựa chọn hiển thị biến (mặc định chọn biến mục tiêu và 3 biến đầu tiên)
    available_cols = [col for col in [target] + features if col in df_raw.columns]
    selected_features = st.multiselect(
        "Chọn các biến muốn trực quan hóa (Tối đa nên chọn 4 biến để cân đối):",
        options=available_cols,
        default=available_cols[:4]
    )
    
    if not selected_features:
        st.warning("Vui lòng chọn ít nhất một biến để hiển thị đồ thị.")
    else:
        # Tạo lưới hiển thị 2x2 tự động dựa trên số lượng chọn
        cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with cols[idx % 2]:
                st.write(f"**Biểu đồ biến: {col_name}**")
                if col_name == target or df_raw[col_name].nunique() <= 5:
                    # Biến phân loại hoặc biến mục tiêu nhị phân
                    fig = px.bar(
                        df_raw[col_name].value_counts().reset_index(),
                        x='index' if 'index' in df_raw[col_name].value_counts().reset_index().columns else df_raw[col_name].value_counts().reset_index().columns[0],
                        y='count' if 'count' in df_raw[col_name].value_counts().reset_index().columns else df_raw[col_name].value_counts().reset_index().columns[1],
                        labels={'index': col_name, 'count': 'Số lượng'},
                        color_discrete_sequence=['#1f77b4'],
                        height=300
                    )
                else:
                    # Biến liên tục
                    fig = px.histogram(
                        df_raw, x=col_name, 
                        marginal="box", 
                        color_discrete_sequence=['#2ca02c'],
                        height=300
                    )
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ---------------------------------------------------------
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng mô hình phân loại")
    
    if 'results' not in st.session_state:
        st.info("💡 Chưa có dữ liệu huấn luyện hiện tại. Vui lòng thiết lập cấu hình ở Sidebar và bấm nút **🚀 Huấn luyện mô hình**.")
    else:
        res = st.session_state['results']
        
        # Chỉ tiêu vô hướng dạng thẻ điểm metric
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{res['accuracy']:.4f}")
        m_col2.metric("Precision", f"{res['precision']:.4f}")
        m_col3.metric("Recall (Sensitivity)", f"{res['recall']:.4f}")
        m_col4.metric("F1-Score", f"{res['f1']:.4f}")
        
        st.divider()
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.write("##### 🧮 Ma trận nhầm lẫn (Confusion Matrix):")
            cm = res['cm']
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                x=['Hợp pháp (0)', 'Gian lận/Mặc định (1)'],
                y=['Hợp pháp (0)', 'Gian lận/Mặc định (1)'],
                color_continuous_scale='Blues',
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_right:
            st.write("##### 📋 Báo cáo chi tiết (Classification Report):")
            df_report = pd.DataFrame(res['report']).transpose()
            st.dataframe(df_report.style.format(precision=4), use_container_width=True)

# ---------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ---------------------------------------------------------
with tab4:
    st.subheader("🔮 Chẩn đoán & Dự báo rủi ro thực tế")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng huấn luyện mô hình thành công tại Sidebar trước khi thực hiện chức năng dự báo.")
    else:
        model = st.session_state['trained_model']
        
        mode = st.radio(
            "Phương thức nhập dữ liệu kiểm tra đầu vào:",
            options=["Chế độ 1: Nhập trực tiếp thủ công", "Chế độ 2: Tải file dữ liệu tổng hợp (X_new)"],
            horizontal=True
        )
        
        # --- CHẾ ĐỘ 1: NHẬP THỦ CÔNG ---
        if "Chế độ 1" in mode:
            st.write("##### 🛠️ Điền thông số của đối tượng cần kiểm tra:")
            
            # Tạo form nhập dữ liệu động dựa trên 14 đặc trưng
            with st.form("single_prediction_form"):
                form_cols = st.columns(4)
                input_data = {}
                
                # Tính toán giá trị mặc định (trung vị) từ tập dữ liệu thô ban đầu để điền sẵn cho người dùng
                for idx, feat in enumerate(features):
                    col_target = form_cols[idx % 4]
                    default_val = float(df_raw[feat].median()) if feat in df_raw.columns else 0.0
                    min_val = float(df_raw[feat].min()) if feat in df_raw.columns else -100.0
                    max_val = float(df_raw[feat].max()) if feat in df_raw.columns else 100.0
                    
                    input_data[feat] = col_target.number_input(
                        f"Giá trị {feat}",
                        value=default_val,
                        min_value=min_val,
                        max_value=max_val,
                        format="%.6f"
                    )
                
                submit_predict = st.form_submit_button("🎯 Tiến hành dự báo rủi ro")
                
            if submit_predict:
                # Chuyển đổi dict thành DataFrame cấu trúc chuẩn 1 dòng giống lúc train
                df_input = pd.DataFrame([input_data])[features]
                
                # Thực hiện dự báo
                pred_class = model.predict(df_input)[0]
                pred_proba = model.predict_proba(df_input)[0][1]
                
                st.divider()
                st.write("#### 📝 Kết quả phân tích đối tượng:")
                
                p_col1, p_col2 = st.columns(2)
                if pred_class == 1:
                    p_col1.error("🚨 KẾT LUẬN: Giao dịch có dấu hiệu GIAN LẬN / RỦI RO CAO")
                else:
                    p_col1.success("✅ KẾT LUẬN: Giao dịch AN TOÀN / HỢP PHÁP")
                    
                p_col2.metric("Xác suất rủi ro (Probability)", f"{pred_proba * 100:.2f}%")

        # --- CHẾ ĐỘ 2: TẢI FILE BATCH PROCESSING ---
        elif "Chế độ 2" in mode:
            st.write("##### 📂 Dự báo hàng loạt cho danh sách khách hàng mới")
            new_file = st.file_uploader(
                "Tải lên tệp chứa các đặc trưng đầu vào (Ví dụ: X_new.xlsx hoặc CSV)",
                type=["csv", "xlsx", "xls"],
                key="batch_predict_uploader"
            )
            
            if new_file is not None:
                df_new = load_data(new_file.getvalue(), new_file.name)
                
                if df_new is not None:
                    # Kiểm tra xem có đủ 14 cột đặc trưng không
                    missing_feats = [f for f in features if f not in df_new.columns]
                    
                    if missing_feats:
                        st.error(f"❌ Bản ghi tải lên không hợp lệ. Thiếu các cột đặc trưng bắt buộc sau: {missing_feats}")
                    else:
                        # Lấy đúng thứ tự cột để đưa vào mô hình dự đoán
                        df_new_features = df_new[features]
                        
                        # Thực hiện dự báo hàng loạt
                        batch_preds = model.predict(df_new_features)
                        batch_probas = model.predict_proba(df_new_features)[:, 1]
                        
                        # Thêm kết quả vào DataFrame hiển thị
                        df_output = df_new.copy()
                        df_output['Dự_Báo_Default'] = batch_preds
                        df_output['Xác_Suất_Rủi_Ro'] = batch_probas
                        
                        st.success(f"⚡ Đã xử lý xong dữ liệu cho {df_output.shape[0]} bản ghi!")
                        
                        # Thống kê nhanh kết quả hàng loạt
                        risk_count = int(np.sum(batch_preds == 1))
                        safe_count = int(np.sum(batch_preds == 0))
                        
                        stat_c1, stat_c2 = st.columns(2)
                        stat_c1.metric("Số lượng rủi ro phát hiện", f"{risk_count} dòng", f"Tỷ lệ: {(risk_count/len(batch_preds))*100:.2f}%", delta_color="inverse")
                        stat_c2.metric("Số lượng an toàn", f"{safe_count} dòng")
                        
                        st.write("##### 📄 Bảng dữ liệu kết quả chi tiết:")
                        st.dataframe(df_output, use_container_width=True)
                        
                        # Xuất file kết quả dự báo ra định dạng CSV mã hóa utf-8-sig
                        csv_data = df_output.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="Ket_Qua_Du_Bao_Rui_Ro.csv",
                            mime="text/csv"
                        )
