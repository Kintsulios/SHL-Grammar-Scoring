# SHL-Grammar-Scoring
Grammar scoring engine for spoken English using Wav2Vec2 embeddings and Ridge regression.
A speech-based grammar scoring system developed as part of the SHL Research Engineer hiring assessment.

## Problem

The objective is to predict a grammar proficiency score from spoken English audio samples. Each audio recording is approximately 45–60 seconds long, with grammar scores ranging from 0 to 5 in the provided training data.

The competition evaluates predictions using:

* Pearson Correlation
* Root Mean Squared Error (RMSE)

## Approach

The implemented pipeline uses:

1. **Audio preprocessing**

   * Audio files are loaded at 16 kHz.
   * Mono audio is used for consistency.

2. **Baseline acoustic features**

   * Duration
   * RMS energy statistics
   * Zero-crossing rate statistics
   * Random Forest regression was used as an initial baseline.

3. **Pretrained speech representations**

   * `facebook/wav2vec2-base` is used to extract 768-dimensional speech embeddings.
   * Mean pooling over the temporal dimension produces a fixed-length representation for each recording.

4. **Regression**

   * Standardized Wav2Vec2 embeddings are passed to Ridge regression.
   * Ridge regularization parameter was tuned using 5-fold cross-validation.
   * The selected value was `alpha = 100`.

5. **Prediction constraints**

   * Final predictions are clipped to the valid score range of 0–5.

## Validation Results

Using 5-fold cross-validation:

| Model                             |       RMSE |    Pearson |
| --------------------------------- | ---------: | ---------: |
| Acoustic features + Random Forest |     0.9058 |     0.6825 |
| Wav2Vec2 + Ridge                  |     0.7232 |     0.8137 |
| Wav2Vec2 + Ridge (`alpha=100`)    | **0.6951** | **0.8277** |

The final model's training RMSE on the full training set was **0.4927**.

## Competition Result

Initial competition submission:

* Leaderboard score: **0.5718**
* Leaderboard rank: **232**

The model is being further investigated and improved as part of the assessment.

## Repository Contents

* `grammar_scoring.ipynb` — Kaggle notebook containing the complete analysis, feature extraction, modelling, evaluation, and submission pipeline.

## Notes

The competition dataset and audio files are not included in this repository because they are provided through the private Kaggle competition.

## Tools

* Python
* NumPy
* Pandas
* Librosa
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* SciPy
