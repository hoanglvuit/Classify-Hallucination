# UIT_Champion - DSC 2025 Track B: Phân loại Hallucination

Đây là source code của đội **UIT_Champion** tham dự cuộc thi **UIT Data Science Challenge 2025 - Track B - Phân loại Hallucination**.

## 📋 Về cuộc thi

**UIT Data Science Challenge 2025** tập trung vào việc đánh giá độ tin cậy của các Mô hình Ngôn ngữ Lớn (LLM) với tiếng Việt, đặc biệt là khả năng phát hiện **Hallucination** (Ảo giác ngôn ngữ).

### Nhiệm vụ chính

Phân loại đầu ra (generated response) của LLM thành 3 nhãn dựa trên so sánh với ngữ cảnh (context) và câu hỏi (prompt):

- **`no`**: Không ảo giác, phản hồi hoàn toàn phù hợp và chỉ dựa vào ngữ cảnh
- **`intrinsic`**: Phản hồi mâu thuẫn hoặc bóp méo thông tin so với ngữ cảnh
- **`extrinsic`**: Phản hồi bổ sung thông tin không có căn cứ hoặc không thể truy xuất từ ngữ cảnh

Tham khảo thêm tại: [CodaBench Competition](https://www.codabench.org/competitions/10153/#/pages-tab)

---

## 📁 Cấu trúc dự án

```
DSC_2025/
│
├── data/                          # Dữ liệu đã xử lý (CSV)
│   ├── train_dsc.csv              # Dữ liệu huấn luyện
│   ├── public_test.csv            # Dữ liệu test công khai
│   └── private_test.csv           # Dữ liệu test riêng tư
│
├── ori_data/                      # Dữ liệu gốc (Excel)
│   ├── vihallu-train.xlsx
│   ├── vihallu-public-test.xlsx
│   └── vihallu-private-test.xlsx
│
├── output/                        # Predictions từ các base models (tên model đầy đủ)
│   ├── cross-encoder_nli-deberta-v3-large/
│   │   └── fold_{0..4}/
│   │       ├── dev_predictions_with_probs.csv
│   │       ├── submit_with_probs_privatetest.csv
│   │       └── ...
│   ├── dangvantuan_vietnamese-document-embedding/
│   ├── FacebookAI_roberta-large-mnli/
│   ├── microsoft_deberta-xlarge-mnli/
│   ├── SemViQA_tc-erniem-viwikifc/
│   └── SemViQA_tc-xlmr-isedsc01/
│
│
├── results/                       # Checkpoints của các mô hình đã huấn luyện
│   └── model_{model_name}/
│       └── fold_{0..4}/
│
├── Vietnamese_impl/               # Custom implementation cho Vietnamese model
│   ├── __init__.py
│   ├── configuration.py           # VietnameseConfig
│   └── modeling.py                # VietnameseForSequenceClassification
│
├── train.py                       # Script huấn luyện chính
├── inference.py                   # Script inference từ checkpoint
├── stack_ensemble.py              # Stack ensemble với XGBoost (inference)
├── translate_data.py              # Dịch dữ liệu từ tiếng Việt sang tiếng Anh
├── utils.py                       # Các hàm tiện ích (preprocessing, evaluation, ...)
├── test.py                        # Soft voting ensemble
├── XGBoost.ipynb                  # Notebook train XGBoost với GridSearchCV
│
├── train.sh                       # Script huấn luyện tất cả models
├── inference.sh                   # Script inference tất cả models
├── requirements.txt               # Dependencies
├── xgb_best_model.pkl            # XGBoost model đã được train (best từ GridSearch)
└── README.md                      # File này
```

---

## 🎯 Ý tưởng Pipeline

### 1. **Kiến trúc tổng thể: Ensemble Learning với Stacking**

Pipeline sử dụng phương pháp **Stacking Ensemble** với 2 tầng:

- **Tầng 1 (Base Models)**: 6 mô hình khác nhau được huấn luyện độc lập
- **Tầng 2 (Meta Model)**: XGBoost kết hợp predictions từ tất cả base models

### 2. **Base Models (Tầng 1)**

Sử dụng 6 mô hình đa dạng để tận dụng các kiến trúc và cách tiếp cận khác nhau:

| Model | Kiến trúc | Ngôn ngữ | Đặc điểm |
|-------|-----------|----------|----------|
| `microsoft/deberta-xlarge-mnli` | DeBERTa-XLarge | EN | Pre-trained trên NLI, mạnh về reasoning |
| `cross-encoder/nli-deberta-v3-large` | Cross-Encoder DeBERTa-v3 | EN | Tối ưu cho NLI tasks |
| `dangvantuan/vietnamese-document-embedding` | Vietnamese Custom | VI | Chuyên biệt cho tiếng Việt |
| `FacebookAI/roberta-large-mnli` | RoBERTa-Large | EN | Pre-trained trên NLI |
| `SemViQA/tc-erniem-viwikifc` | ERNIE-M | VI | Claim-based model cho fact-checking |
| `SemViQA/tc-xlmr-isedsc01` | XLM-RoBERTa | VI | Multilingual, pre-trained cho fact-checking |

### 3. **Xử lý dữ liệu**

- **Dịch thuật**: Sử dụng `VietAI/envit5-translation` để dịch dữ liệu tiếng Việt sang tiếng Anh cho các mô hình EN
- **Tokenization**: 
  - Mô hình EN: `[context_en]` + `[response_en]`
  - Mô hình VI: `[context_vi]` + `[response_vi]`
- **Cross-Validation**: 5-fold Stratified K-Fold để đảm bảo phân phối nhãn đều

### 4. **Stack Ensemble (Tầng 2)**

- **Input Features**: Xác suất từ 6 base models (18 features: `prob_intrinsic`, `prob_extrinsic`, `prob_no` × 6 models)
- **Meta Model**: XGBoost Classifier
- **Training**: 
  - Sử dụng OOF (Out-of-Fold) predictions từ base models để train meta model
  - **GridSearchCV** để tìm best hyperparameters:
    - `n_estimators`: [300, 500, 700]
    - `max_depth`: [3, 6]
    - `learning_rate`: [0.01, 0.1, 0.2]
    - `subsample`: [0.8, 1]
    - `colsample_bytree`: [0.8, 1]
    - `gamma`: [0, 0.1, 0.5]
  - Cross-validation: 5-fold Stratified K-Fold
  - Scoring metric: F1 Macro

---

## 🔄 Quy trình thực hiện

### Bước 1: Chuẩn bị dữ liệu

```bash
# Dịch dữ liệu từ tiếng Việt sang tiếng Anh
python translate_data.py
```

- Đọc dữ liệu từ `ori_data/*.xlsx`
- Dịch `context`, `prompt`, `response` sang tiếng Anh
- Lưu vào `data/*.csv` với format: `context_vi`, `context_en`, `prompt_vi`, `prompt_en`, `response_vi`, `response_en`

### Bước 2: Huấn luyện Base Models

```bash
chmod +x ./train.sh
./train.sh
```

Quy trình cho mỗi model:

1. **Load và preprocess dữ liệu** (`utils.py::preparing_dataset`):
   - Đọc CSV, chuyển sang Dataset format
   - Map labels: `intrinsic`, `extrinsic`, `no` → số nguyên
   - Load tokenizer

2. **Cross-Validation Training** (`train.py::ensemble_training`):
   - Chia dữ liệu thành 5 folds (Stratified K-Fold)
   - Với mỗi fold:
     - Tokenize: `[context]` + `[response]` (hoặc `[prompt] + [context]` + `[response]`)
     - Fine-tune model với Transformers Trainer
     - Evaluate trên dev set
     - Predict trên test sets (public & private)
     - Lưu predictions với xác suất vào `output/{model_name}/fold_{i}/`

3. **Các mô hình được train**:
   - DeBERTa-XLarge (EN, 1 epoch)
   - Cross-Encoder DeBERTa-v3 (EN, 2 epochs)
   - Vietnamese Document Embedding (VI, 4 epochs)
   - RoBERTa-Large (EN, 2 epochs)
   - ERNIE-M (VI, 3 epochs)
   - XLM-RoBERTa (VI, 3 epochs)

### Bước 3: Inference (nếu cần)

```bash
chmod +x ./inference.sh
TRANSLATE=false ./inference.sh  # Bỏ qua bước dịch
```

- Load checkpoint từ `results/`
- Predict trên private test set
- Lưu predictions vào `output/`

### Bước 4: Feature Engineering cho Stack Ensemble

(`utils.py::feature_engineering`):

- Thu thập OOF predictions từ tất cả folds của mỗi model
- Với train: Stack tất cả folds theo chiều dọc
- Với test: Average xác suất qua 5 folds
- Tạo feature matrix: 18 features (3 probs × 6 models)

### Bước 5: Train XGBoost Meta Model (GridSearchCV)

**Sử dụng notebook `XGBoost.ipynb`**:

1. **Feature Engineering**:
   - Thu thập OOF predictions từ tất cả folds của 6 base models
   - Tạo feature matrix: 18 features (3 probs × 6 models)
   - Với train: Stack tất cả folds theo chiều dọc
   - Với test: Average xác suất qua 5 folds

2. **GridSearchCV**:
   - Tìm best hyperparameters với 5-fold Stratified K-Fold
   - Scoring: F1 Macro
   - Tổng cộng 216 candidates (6 × 2 × 3 × 2 × 2 × 3)
   - Seed: `22520465` (reproducibility)

3. **Lưu model**:
   - Best model được lưu vào `xgb_best_model.pkl`

**Lưu ý**: Notebook này được chạy trên Kaggle/môi trường có GPU để tăng tốc GridSearch.

### Bước 6: Inference với XGBoost Model

```bash
pip install scikit-learn==1.2.2 xgboost
python stack_ensemble.py
```

- Load XGBoost model đã được train (`xgb_best_model.pkl`)
- Predict trên test features
- Output: `submit.csv` với format `id, predict_label`

---


## 🚀 Hướng dẫn reproduce kết quả

### Cách 1: Chỉ chạy Stack Ensemble (nhanh nhất)

Giả sử đã có sẵn predictions từ các base models:

```bash
pip install scikit-learn==1.2.2 xgboost
python stack_ensemble.py
```

Kết quả: `submit.csv`

### Cách 2: Inference từ checkpoint (không train lại)

```bash
chmod +x ./inference.sh
TRANSLATE=false ./inference.sh
python stack_ensemble.py --root_folder "output" --use_true False
```

### Cách 3: Reproduce toàn bộ (train + inference + ensemble)

```bash
chmod +x ./train.sh
./train.sh
```

Sau đó train XGBoost model (nếu chưa có `xgb_best_model.pkl`):

1. Mở notebook `XGBoost.ipynb`
2. Chạy tất cả cells để train với GridSearchCV
3. Model best sẽ được lưu vào `xgb_best_model.pkl`

Cuối cùng, chạy inference:

```bash
python stack_ensemble.py
```

**Lưu ý**: Nếu đã có `xgb_best_model.pkl`, có thể bỏ qua bước train XGBoost và chạy trực tiếp `stack_ensemble.py`

---

## 📊 Kết quả

- Predictions cuối cùng được lưu trong `submit.csv`
- Format: `id, predict_label` (với `predict_label` là `no`, `intrinsic`, hoặc `extrinsic`)

---

## 🔧 Dependencies

Xem `requirements.txt` để biết chi tiết. Các thư viện chính:

- `transformers`: 4.57.0
- `torch`: PyTorch
- `scikit-learn`: 1.7.2
- `pandas`: 2.3.3
- `xgboost`: Cho stack ensemble
- `evaluate`: 0.4.6
- `openpyxl`: 3.1.5 (đọc Excel)

---

## 👥 Đội thi

**UIT_Champion** - UIT Data Science Challenge 2025
