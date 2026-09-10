# Voice Clone Detection

AI-powered prototype for detecting AI-generated and cloned voices.

## Problem

Voice cloning technology can be abused to impersonate individuals and
perform social engineering, financial fraud, identity theft, and other
security attacks.

This project aims to analyze an audio recording and estimate whether the
voice is likely to be real or AI-generated/cloned.

---

## Architecture

```text
Audio
  ↓
Audio Validation
  ↓
Preprocessing
  ↓
MFCC + Acoustic Features
  ↓
Machine Learning Classifier
  ↓
REAL / CLONED
  ↓
Confidence + Risk Score
  ↓
FastAPI
  ↓
Frontend