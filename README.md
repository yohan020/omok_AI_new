# AlphaZero実装: 8x8 五目並べ強化学習AI (AlphaZero Implementation: 8x8 Gomoku AI)

## 1. 概要 (Overview)
AlphaGo Zeroのアルゴリズム（MCTS + ResNet）をベースに、人間の棋譜データを一切使用せず、自己対局（Self-Play）のみで学習する五目並べAIを実装しました。
強化学習特有の課題（報酬の希薄性、局所解への停滞）に対し、独自の工夫（中間報酬、仮想フェンスなど）を導入して解決を図りました。

最終的に、開発者（人間）との対局において、定石通りの中央戦では「理論的な引き分け（Nash Equilibrium）」に達するレベルの防御力を実現しました。

## 2. 技術スタック (Tech Stack)
* **Language:** Python 3.10
* **Framework:** PyTorch (Deep Learning), NumPy
* **GUI:** Tkinter
* **Algorithm:**
    * Monte Carlo Tree Search (MCTS)
    * ResNet (Residual Neural Network) - Policy & Value Network

## 3. 主要な技術的工夫 (Key Engineering Challenges & Solutions)

学習過程で直面した課題に対し、以下のエンジニアリング的アプローチを行いました。

### 🛡️ 1. 初期学習の効率化: Virtual Fence (仮想フェンス)
* **課題:** 初期のAIが盤面の隅（コーナー）に無意味な着手をする傾向があり、学習効率が悪かった。
* **解決:** 学習初期の一定手数（6手）まで、着手範囲を中央4x4に制限する「仮想フェンス」を導入。
* **結果:** 有意な戦闘データ（中央での攻防）を効率的に収集させ、学習収束速度を大幅に向上させた。

### ⚔️ 2. 攻撃性の向上: Intermediate Reward (中間報酬)
* **課題:** 勝敗のみを報酬（+1/-1）にすると、AIが「負けないこと」を最優先し、攻撃をしない「過度な守備的AI」になってしまった。
* **解決:** 勝敗が決まる前でも、「3連（Open 3）」や「4連（Open 4）」を作った時点で即時報酬（+0.2, +0.5）を与える仕組みを導入。
* **結果:** AIが自発的に攻撃パターンを構築するようになり、攻撃と防御のバランスが取れたモデルへと進化した。

### 🔄 3. 局所解からの脱出: SGDR (学習率リセット)
* **課題:** 学習が進むにつれて特定の戦術に固執し、成長が停滞する「局所最適解（Local Optima）」に陥った。
* **解決:** 学習率（Learning Rate）が下限に達した際、周期的に初期値へリセットするSGDR (Stochastic Gradient Descent with Warm Restarts) を適用。
* **結果:** モデルに強制的な「探索（Exploration）」を促し、より汎用的な打ち手を学習させることに成功した。

## 4. 成果と考察 (Results & Insights)

### ✅ 成果: 鉄壁の防御
約1,000サイクルの学習後、AIは中央からの展開において人間がどのような攻撃を仕掛けても的確に防御し、**「引き分け（Draw）」**に持ち込む実力を獲得しました。

### ⚠️ 発見: AIの死角 (Blind Spot)
完成したAIに対し、学習データに含まれない「盤面の隅（学習範囲外）」から奇襲攻撃を仕掛けたところ、AIが対応できずに敗北する現象を確認しました。
* **考察:** 「仮想フェンス」による効率化が、逆に「分布外データ（Out-of-Distribution）」への対応力を弱めるというトレードオフ（過学習）を確認しました。これは、AlphaGoにおける「神の一手（第78手）」と同様の現象であり、AIの汎化能力における重要な知見を得ることができました。

## 5. 実行方法 (Usage)

### 必須ライブラリのインストール
```bash
pip install torch numpy tqdm
```
### 学習の実行
```bash
python train_mcts.py
```
### 対局（GUI）
```bash
python play_gui_non_hybrid.py
```
