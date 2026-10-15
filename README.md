# SHL Grammar Scoring Engine

An audio-based machine learning project for predicting spoken English grammar scores from speech recordings, developed for the **SHL Hiring Assessment 2026** competition.

The objective is to predict a continuous grammar score between **0 and 5** from audio samples approximately 45–60 seconds long. The project evolved from a baseline speech-embedding regression model into a more advanced multimodal pipeline combining speech representations, transcription-based linguistic features, and model ensembling.

## Project Evolution: Baseline → Advanced Pipeline

The project contains two notebooks representing different stages of experimentation.

| Area | Original notebook | Improved notebook |
|---|---|---|
| Notebook | `shl-grammar-scoring-engine.ipynb` | `shl_grammar_scoring_whisper_wavlm.ipynb` |
| Speech representation | Wav2Vec2 embeddings | WavLM and Whisper encoder representations |
| Audio understanding | Primarily audio embeddings and acoustic features | Speech embeddings combined with transcription-derived information |
| Language features | Limited linguistic information | Whisper transcriptions, E5 text embeddings, and handcrafted text features |
| Model strategy | Ridge regression and Random Forest baselines | RidgeCV, SVR, gradient boosting, blending, and stacking experiments |
| Main objective | Establish a reliable baseline | Improve predictive performance by combining complementary signals |

The second notebook builds on the first rather than replacing it. It explores whether speech representations, linguistic information, and model combinations can capture different aspects of grammar proficiency more effectively than a single-modality baseline.

## 1. Original Notebook: Establishing the Baseline

The first notebook established a benchmark using speech representations and conventional machine learning.

### What it implemented

- Extracted **Wav2Vec2 speech embeddings** to represent information learned from audio.
- Used handcrafted acoustic features with a Random Forest regression model.
- Trained Ridge regression on speech embeddings to predict grammar scores.
- Compared model performance using RMSE and Pearson correlation.
- Evaluated the effect of changing the Ridge regularization parameter.

### Baseline validation results

| Model | RMSE ↓ | Pearson correlation ↑ |
|---|---:|---:|
| Acoustic features + Random Forest | 0.9058 | 0.6825 |
| Wav2Vec2 + Ridge | 0.7232 | 0.8137 |
| Wav2Vec2 + Ridge, alpha = 100 | **0.6951** | **0.8277** |

Lower RMSE indicates more accurate predictions, while higher Pearson correlation indicates a stronger relationship between predicted and actual scores.

**Key finding:** Wav2Vec2 embeddings performed substantially better than the handcrafted acoustic-feature baseline. Increasing Ridge regularization also improved the reported validation results.

However, the baseline primarily relied on audio representations. It did not fully exploit the complementary information available from speech transcription, text semantics, and more diverse model architectures.

## 2. Improved Notebook: A Multimodal, Ensemble-Based Approach

The second notebook extends the baseline by investigating several complementary sources of information from each speech recording.

### What improved

**1. Added Whisper transcription**

Whisper large-v3 is used to transcribe speech in chunks. This introduces textual information that can help characterize grammar and sentence structure rather than relying exclusively on audio embeddings.

The pipeline also extracts transcription-related information, including confidence-oriented and log-probability/entropy features.

**2. Expanded speech representations**

Instead of depending only on Wav2Vec2, the improved pipeline explores:

- **WavLM-large embeddings** to represent speech characteristics.
- **Whisper encoder embeddings** to capture additional learned audio representations.
- **E5 text embeddings** to represent the semantic information in transcribed speech.

These representations provide complementary perspectives on the same recording.

**3. Added handcrafted linguistic features**

The improved pipeline incorporates text-derived features alongside learned embeddings. This allows the models to use both high-dimensional learned representations and explicitly calculated linguistic signals.

**4. Broadened model experimentation**

The improved notebook investigates multiple regression approaches, including:

- RidgeCV
- Support Vector Regression (SVR)
- Histogram-based gradient boosting
- Blending and stacking of model predictions

Rather than assuming one model is best, the pipeline tests whether combining different models and feature representations can improve generalization.

**5. Investigated layer-wise representations and stacking**

Further experiments examine additional WavLM and Whisper encoder layers and compare different combinations in a stacking setup. These experiments help assess whether representations from different model layers provide useful complementary information.

## 3. Results and Progress

### Baseline versus advanced experiments

| Experiment | RMSE ↓ | Pearson correlation ↑ |
|---|---:|---:|
| Original notebook: Wav2Vec2 + Ridge, alpha = 100 | 0.6951 | 0.8277 |
| Improved pipeline: WavLM + Ridge, OOF | 0.5200 | 0.8585 |
| Improved pipeline: Whisper encoder + Ridge, OOF | 0.5285 | 0.8539 |
| Advanced stacking reference | 0.4896 | 0.8758 |
| Exploratory all-layer blend | **0.4885** | **0.8764** |

**Important evaluation note:** The original baseline validation metrics and the advanced out-of-fold (OOF) results come from different evaluation procedures and should not be treated as a perfectly controlled, apples-to-apples comparison. The table shows the progression of experiments, not a definitive estimate of the improvement under an identical evaluation protocol.

The all-layer result is an **exploratory cross-validation experiment**. It was not used to regenerate the saved submission file, so the 0.4885 RMSE should not be interpreted as the performance of that submission.

### What the experiments suggest

- WavLM and Whisper encoder representations both provide useful predictive signals.
- Combining multiple representations and model predictions can outperform the individual representation-based experiments in the reported advanced evaluation.
- The exploratory all-layer experiment produced a small improvement over the stacking reference, motivating further investigation.
- The next step is to validate the final ensemble consistently and evaluate a regenerated submission on the competition leaderboard.

## 4. Competition Submission

The original competition submission achieved:

- **Public leaderboard score:** 0.5718
- **Reported rank:** 232

The improved notebook generates `submission_v3.csv`. Its final competition leaderboard score has not yet been established, so no leaderboard improvement is claimed for the new submission.

## 5. Repository Structure

```text
SHL-Grammar-Scoring/
├── README.md
├── shl-grammar-scoring-engine.ipynb
└── shl_grammar_scoring_whisper_wavlm.ipynb
```

- `shl-grammar-scoring-engine.ipynb` — Original baseline and initial experiments.
- `shl_grammar_scoring_whisper_wavlm.ipynb` — Improved transcription, representation learning, feature engineering, and ensemble experiments.
- `README.md` — Project overview, methodology, and reported results.

## 6. Technology Stack

- **Language:** Python
- **Audio processing:** Librosa
- **Deep learning:** PyTorch
- **Speech models:** Hugging Face Transformers, Whisper large-v3, WavLM-large
- **Text representations:** E5 embeddings
- **Machine learning:** Scikit-learn
- **Numerical computing:** NumPy, Pandas, SciPy
- **Environment:** Kaggle notebooks with GPU acceleration

## 7. Running the Notebooks

The notebooks were developed for the Kaggle competition environment.

The training dataset is not included in this repository. To reproduce the experiments:

1. Access the SHL Hiring Assessment 2026 competition dataset on Kaggle.
2. Enable suitable GPU acceleration. The improved notebook is configured for a multi-GPU workflow.
3. Install or enable the required Python libraries.
4. Run the notebook cells in order, allowing the required pretrained models to be downloaded.
5. Execute the submission-generation cells to create the prediction CSV.

The improved pipeline is computationally more demanding than the baseline because it uses large pretrained speech models and additional feature extraction.

## 8. Key Takeaway

This project demonstrates an iterative machine learning workflow: establish a baseline, identify opportunities to improve the feature space, experiment with complementary pretrained representations, compare regression models, and investigate ensemble methods using cross-validation.

The central progression is from **single-representation speech regression** to a **multimodal speech-and-text pipeline with model blending and stacking**. The work also highlights the importance of separating cross-validation results, exploratory experiments, and actual competition leaderboard performance.

