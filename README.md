# Robust Image Similarity for Noisy Earth Observations

This project implements a feature-based image similarity pipeline designed to compare satellite or aerial images even under significant noise or distortion. It uses traditional computer vision techniques such as Gaussian blurring, ORB feature detection, and Hamming-distance matching to generate a normalized similarity score between image pairs.

## 🔍 Problem Statement

Given a set of Earth observation images, some of which are noisy or overlapping, the goal is to identify pairs that originate from the same geographic region and compute a similarity score in the range [0, 1].

## 📌 Key Features

- Gaussian blur preprocessing to handle high-frequency noise
- ORB (Oriented FAST and Rotated BRIEF) for keypoint detection
- Brute-force feature matching with Hamming distance
- Normalized similarity scoring independent of feature count
- Lightweight and does not rely on deep learning models

## 📂 File Structure

