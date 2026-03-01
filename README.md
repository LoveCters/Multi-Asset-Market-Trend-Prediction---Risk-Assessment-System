# Hệ thống Dự báo Xu hướng và Đánh giá Rủi ro Thị trường Đa tài sản
*(Multi-Asset Market Trend Prediction & Risk Assessment System)*

## Tổng quan dự án (Project Overview)
Dự án này là một hệ thống học sâu (deep learning pipeline) toàn diện được thiết kế để dự báo xu hướng thị trường chứng khoán (Tăng/Giảm) trong khung thời gian 10 ngày tới, đồng thời đánh giá rủi ro giao dịch trên một danh mục gồm 10 loại tài sản đa dạng.

Khác với các mô hình dự báo truyền thống chỉ đưa ra một giá trị dự đoán duy nhất (dễ thất bại trong môi trường tài chính nhiều nhiễu), hệ thống này sử dụng phương pháp tiếp cận Nhận thức Trạng thái thị trường (Regime-Aware) kết hợp với Hồi quy phân vị (Quantile Regression). Hệ thống không chỉ dự báo hướng đi của giá mà còn cung cấp khoảng tin cậy 80% để định lượng mức độ không chắc chắn, hỗ trợ nhà giao dịch thiết lập các mức Cắt lỗ (Stop-Loss) và Chốt lời (Take-Profit) linh hoạt.

## Tính năng chính (Key Features)
* Mô hình hóa toàn cục đa tài sản (Multi-Asset Global Modeling): Huấn luyện một mô hình duy nhất đồng thời trên 10 mã chứng khoán (SPY, QQQ, IWM, TLT, GLD, NVDA, JPM, XOM, JNJ, VNQ) bằng cách sử dụng entity embeddings để nắm bắt cả tương quan thị trường chung lẫn đặc tính riêng của từng tài sản.
* Nhận thức ngữ cảnh thông qua HMM (Contextual Awareness via HMM): Ứng dụng Mô hình Markov ẩn (Hidden Markov Model - HMM) để tự động phân loại thị trường thành 3 trạng thái (Tăng trưởng, Suy thoái, Đi ngang). Tính năng này hoạt động như một bộ lọc vĩ mô cho các tín hiệu giao dịch.
* Đánh giá rủi ro với TFT (Risk Assessment with TFT): Sử dụng kiến trúc hiện đại Temporal Fusion Transformer (TFT) với hàm mất mát Quantile Loss (Q10, Q50, Q90) để tạo ra các dự báo đã được hiệu chuẩn rủi ro.
* Ngưỡng quyết định động (Dynamic Thresholding): Triển khai kỹ thuật xác định ngưỡng động (dựa trên trung vị của các dự báo) để chuyển đổi đầu ra hồi quy liên tục thành các tín hiệu phân loại Tăng/Giảm mạnh mẽ, khắc phục hiệu quả thiên kiến "Bám trung bình" (Mean Reversion).
* Kiểm thử ngược mạnh mẽ (Robust Backtesting): Được đánh giá thông qua phương pháp Kiểm chứng Cửa sổ trượt (Sliding Window Validation) nghiêm ngặt nhằm ngăn chặn rò rỉ dữ liệu (Data Leakage) và đảm bảo tính ổn định của hiệu suất qua các đợt khủng hoảng lịch sử (ví dụ: Khủng hoảng 2008, COVID-19 2020).

## Tập dữ liệu & Đặc trưng (Dataset & Features)
* Tài sản: 10 mã chứng khoán đại diện cho các chỉ số chính, các nhóm ngành và tài sản phòng thủ.
* Khung thời gian: 1990 - 2026 (Dữ liệu ngày).
* Trích xuất đặc trưng (Feature Engineering):
  * Dữ liệu quá khứ (Unknown Future Inputs): Lợi nhuận giá, Chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, ATR, Khoảng cách đến SMA), Xác suất trạng thái HMM (Regime Probabilities).
  * Dữ liệu tương lai đã biết (Known Future Inputs): Đặc trưng lịch (Ngày trong tuần, Tháng).
  * Biến đồng thời tĩnh (Static Covariates): Mã tài sản (Symbol).

## Kiến trúc Mô hình & Phương pháp luận (Model Architecture & Methodology)
Dự án tiến hành đối chuẩn (benchmark) nhiều kiến trúc khác nhau:
1. Mô hình thống kê cơ sở: ARIMAX
2. Học máy truyền thống: XGBoost
3. Học sâu (Mạng nơ-ron hồi quy): LSTM, Bidirectional LSTM (BiLSTM)
4. Mô hình hiện đại (Dựa trên Attention): Temporal Fusion Transformer (TFT)

Quy trình của mô hình TFT:
`Dữ liệu thô` -> `Trích xuất đặc trưng & Nhận diện HMM Regime` -> `TimeSeriesDataSet (PyTorch Forecasting)` -> `Huấn luyện TFT Quantile` -> `Đầu ra hiệu chuẩn rủi ro (Q10, Q50, Q90)` -> `Hậu xử lý Phân loại`

## Kết quả chính & Giá trị thực tiễn (Key Results & Business Value)
Trong ứng dụng học máy vào tài chính, mức Độ chính xác (Accuracy) > 60% thường là dấu hiệu của việc rò rỉ dữ liệu (Data Leakage). Dự án này chủ ý loại bỏ hoàn toàn rò rỉ dữ liệu để mô phỏng điều kiện giao dịch thực tế khắc nghiệt nhất.

* Hiệu suất: Mô hình TFT đạt Độ chính xác tổng thể (Overall Accuracy) khoảng 53%, nhưng quan trọng hơn, đạt Độ chuẩn xác (Precision) xấp xỉ 61% cho các quyết định giao dịch theo định hướng.
* Hiệu chuẩn rủi ro: Mô hình đạt Độ bao phủ (Coverage) xấp xỉ 80% cho các Khoảng tin cậy (Q10-Q90), chứng minh hiệu quả trong việc ước lượng chính xác biến động và độ bất ổn của thị trường.
* Phân tích chuyên sâu: Mô hình chứng minh rằng trong Trạng thái 0 (Thị trường giảm/Suy thoái), việc dựa vào phân vị bi quan Q10 giúp cải thiện đáng kể khả năng bắt đáy (Down-Recall), đóng vai trò như một hệ thống cảnh báo sớm hiệu quả cho các đợt sụp đổ của thị trường.

## Công nghệ sử dụng (Technologies Used)
* Học sâu (Deep Learning): PyTorch, PyTorch Lightning, PyTorch Forecasting
* Học máy (Machine Learning): Scikit-learn, XGBoost, Statsmodels (ARIMAX, HMM)
* Xử lý dữ liệu: Pandas, NumPy, yfinance
* Trực quan hóa: Matplotlib

