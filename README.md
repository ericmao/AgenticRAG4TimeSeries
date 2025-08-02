# Core Agentic RAG for CERT Anomaly Analysis

## 專案概述

這是一個專注於核心 Agentic RAG 功能的專案，用於 CERT 內部威脅數據的異常分析。系統整合了多種異常檢測技術和 GPT-4o 語言模型來提供全面的安全分析。

## 核心功能

### 🔍 多模態異常檢測
- **Markov Chain 檢測器**: 分析用戶行為序列的模式轉換
- **BERT 異常檢測器**: 基於文本序列的深度學習異常檢測
- **向量存儲**: 用於相似行為搜索的語義檢索

### 🤖 GPT-4o 智能分析
- **用戶行為分析**: 結合異常分數和行為特徵的綜合評估
- **模式識別**: 識別潛在的內部威脅模式
- **風險評估**: 提供詳細的風險等級和建議

### 📊 數據處理
- **CERT 數據集**: 處理真實的內部威脅數據
- **特徵工程**: 自動提取時間序列特徵
- **文本化處理**: 將事件轉換為可分析的文本序列

## 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置 OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 運行核心分析

```bash
# 運行核心 Agentic RAG 系統
python scripts/main_agentic_rag_cert.py
```

## 專案結構

```
AgenticRAG/
├── scripts/
│   ├── main_agentic_rag_cert.py   # 核心 Agentic RAG 系統
│   └── main.py                    # 原始主腳本
├── src/
│   ├── core/                      # 核心組件
│   │   ├── data_processor.py      # 數據處理器
│   │   ├── vector_store.py        # 向量存儲
│   │   ├── markov_anomaly_detector.py
│   │   └── bert_anomaly_detector.py
│   ├── utils/                     # 工具函數
│   │   └── model_persistence.py   # 模型持久化
│   └── agents/                    # 代理組件
├── models/                        # 訓練好的模型
├── data/                          # 數據文件
└── docs/                          # 文檔
```

## 核心功能詳解

### 異常檢測流程

1. **數據預處理**
   - 加載 CERT 內部威脅數據集
   - 工程化時間序列特徵
   - 文本化事件序列

2. **模型訓練/加載**
   - 檢查現有訓練好的模型
   - 如果沒有或過期，重新訓練
   - 保存訓練好的模型

3. **異常分析**
   - Markov Chain 分析用戶行為模式
   - BERT 分析文本序列異常
   - 結合分數進行綜合評估

4. **智能分析**
   - GPT-4o 分析異常分數和行為特徵
   - 生成風險評估和建議
   - 提供詳細的安全分析報告

## 技術特點

### 🔧 模型持久化
- 自動檢查和加載現有模型
- 智能決定是否需要重新訓練
- 支持模型版本管理

### 🚀 性能優化
- 向量化數據處理
- 並行異常檢測
- 高效的語義搜索

### 🛡️ 安全分析
- 多維度異常檢測
- 智能風險評估
- 可解釋的分析結果

## 配置選項

### 模型配置
```python
# 在腳本中修改
use_existing, model_status = should_use_existing_models(
    force_retrain=False,        # 是否強制重新訓練
    max_model_age_days=30       # 模型最大年齡（天）
)
```

### LLM 配置
```python
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,            # 創造性 vs 一致性
    max_tokens=1000            # 最大輸出長度
)
```

## 故障排除

### 常見問題

1. **模型加載失敗**
   - 檢查 `models/` 目錄是否存在
   - 確認模型文件完整性
   - 重新訓練模型

2. **API Key 錯誤**
   - 確認 OpenAI API Key 設置正確
   - 檢查網絡連接
   - 驗證 API 配額

3. **數據處理錯誤**
   - 確認 CERT 數據集路徑正確
   - 檢查數據格式
   - 驗證特徵工程步驟

## 開發指南

### 添加新的異常檢測器

1. 創建新的檢測器類
2. 實現 `fit()` 和 `detect_anomaly()` 方法
3. 在 `initialize_agentic_rag_system()` 中集成

### 擴展分析功能

1. 修改 `analyze_user_anomalies()` 函數
2. 添加新的分析維度
3. 更新報告生成邏輯

## 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進這個專案！

## 授權

本專案採用 MIT 授權條款。 