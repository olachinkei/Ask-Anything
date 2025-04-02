# MVBench Evaluation Implementation

このディレクトリには、MVBench評価のための主要な実装ファイルが含まれています。

## ファイル構成

```
src/
├── config.py     # 設定ファイル
├── dataset.py    # データセット実装
├── model.py      # LLaVAモデル実装
└── main.py       # メイン実行スクリプト
```

## 各ファイルの説明

### config.py
設定ファイル。以下の設定を含みます：
- プロジェクト設定（PROJECT_NAME）
- データパス設定（BASE_DATA_DIR, JSON_DIR, VIDEO_DIR）
- データセット設定（DATA_LIST）
- モデル設定（NUM_SEGMENTS, RESOLUTION, INPUT_MEAN, INPUT_STD）
- プロンプト設定（SYSTEM_PROMPT, ANSWER_INSTRUCTION_PROMPT）

### dataset.py
MVBenchデータセットの実装。主な機能：
- ビデオ、GIF、フレームの読み込み
- データの前処理と変換
- データセットの統計情報の表示

### model.py
LLaVAモデルの実装。主な機能：
- モデルの初期化と設定
- プロンプトの生成
- 推論の実行
- Weaveとの統合

### main.py
メイン実行スクリプト。主な機能：
- データセットとモデルの初期化
- 評価ループの実行
- 結果の保存と表示
- タスクごとの精度計算

## 使用方法

1. データの準備：
   - `config.py`の`BASE_DATA_DIR`を適切なパスに設定
   - データディレクトリ構造：
     ```
     data/
     └── MVBench/
         ├── json/     # JSONファイル
         └── video/    # ビデオファイル
     ```

2. 実行：
   ```bash
   python -m src.main
   ```

3. 結果：
   - 結果は`results`ディレクトリに保存
   - `evaluation_results.json`に詳細な結果
   - タスクごとの精度が標準出力に表示

## 注意事項

- 相対インポートを使用しているため、`src`ディレクトリから実行する必要があります
- 必要な依存パッケージがインストールされていることを確認してください
- GPUが利用可能な場合は自動的に使用されます 