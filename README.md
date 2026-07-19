# Student Grade Predictor

This is a simple Streamlit app that predicts a student's final year-end grade using machine learning (Linear Regression). It calculates predictions based on their current mid-term marks and total absences. 

It's designed for teachers and school counselors to sit down with students, play with the sliders, and see exactly how raising a grade or cutting down on absences changes their projected final score.

## Features
* **Choose Your Grading Scale:** Switch between 20, 50, or 100-point systems depending on your school. The app handles all the math automatically.
* **Goal Tracker:** Set a target grade to see a live progress bar showing how close the student is to hitting it.
* **Save Snapshots:** A table at the bottom lets you save different slider setups so you can compare multiple scenarios side-by-side.
* **Auto-Data Backup:** If the `StudentData.csv` file isn't in the folder, the app generates fake data on the spot so the AI can still train and work without crashing.

## Files in this Project
* `analysis.py` - The main Python file with the machine learning model and the Streamlit dashboard layout.
* `StudentData.csv` - The historical dataset used to train the prediction model.
* `README.md` - This guide.

## How to Run It Locally

1. **Clone this repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>