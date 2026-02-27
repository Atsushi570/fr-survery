# Face Recognition Model Benchmark

商用利用可能な顔認識モデル4つの精度をLFWデータセットで比較し、ROC曲線で可視化する。

## 対象モデル

| モデル | ライセンス | Embedding次元 |
|--------|-----------|--------------|
| OpenCV SFace | Apache 2.0 | 128d |
| AuraFace-v1 | Apache 2.0 | 512d |
| Dlib ResNet-34 | Public Domain | 128d |
| DeepFace GhostFaceNet | MIT | 512d |

## パイプライン構成

ベンチマークの公平性を確保するため、顔検出とアライメントを全モデル共通の前処理として統一している。

### 1. 顔検出 — YuNet

共通検出器として [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)（OpenCV Zoo）を使用。軽量なONNXモデルで、各画像から最も信頼度の高い顔を1つ検出し、5点ランドマーク（両目・鼻・両口角）を出力する。

### 2. ランドマークリマップ

YuNetのランドマーク順序 `[right_eye, left_eye, nose, right_mouth, left_mouth]` を insightface標準順 `[left_eye, right_eye, nose, left_mouth, right_mouth]` に並べ替える。

### 3. アライメント — ArcFace標準参照点

insightfaceの `arcface_dst`（112x112用の参照5点座標）に対して `SimilarityTransform`（scikit-image）で変換行列を推定し、`cv2.warpAffine` で顔を正規化する。

- **112x112**: SFace / AuraFace / GhostFaceNet 用
- **150x150**: Dlib ResNet-34 用（112を基準にスケーリング）

### 4. 特徴抽出

各認識モデルはアライメント済みBGR画像のみを受け取り、embedding を出力する。検出器の違いによる精度差を排除し、純粋な認識モデルの比較が可能。

## セットアップ

```bash
uv sync
```

## 使い方

```bash
# モデル事前ダウンロード
uv run python scripts/download_models.py

# ベンチマーク実行
uv run python scripts/run_benchmark.py

# 結果分析 (Jupyter)
uv run jupyter notebook notebooks/analysis.ipynb
```

評価済みモデルの結果は `results/` にキャッシュされ、再実行時はスキップされる。
全モデルを再評価する場合は `rm results/*.json` してから実行。

## 出力

- `results/benchmark_results.json` — 全メトリクス (AUC, EER, TAR@FAR)
- `results/roc_curves.png` — ROC曲線 (線形スケール)
- `results/roc_curves_log.png` — ROC曲線 (対数スケール)
