# Face Recognition Model Benchmark

商用利用可能な顔認識モデル4つの精度をLFWデータセットで比較し、ROC曲線で可視化する。

## 対象モデル

| モデル | ライセンス | Embedding次元 |
|--------|-----------|--------------|
| OpenCV SFace | Apache 2.0 | 128d |
| AuraFace-v1 | Apache 2.0 | 512d |
| Dlib ResNet-34 | Public Domain | 128d |
| DeepFace GhostFaceNet | MIT | 512d |

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
