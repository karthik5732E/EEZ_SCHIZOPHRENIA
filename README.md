🧠 EEG-Based Schizophrenia Detection using Hybrid Machine Learning and CNN

📌 Project Overview
This project focuses on detecting schizophrenia using EEG (Electroencephalogram) brain signal data. The system analyzes brain activity patterns and classifies subjects as schizophrenia patients or healthy individuals.
A hybrid approach is implemented by combining traditional machine learning models with a Convolutional Neural Network (CNN) to improve prediction accuracy. EEG signal features are integrated with clinical attributes such as age and gender for better performance.

🎯 Objectives
Analyze EEG signals to identify schizophrenia-related patterns
Extract statistical features from EEG data
Combine EEG features with clinical data
Train and evaluate multiple machine learning models
Compare model performance to identify the best approach


📁 Dataset
The dataset used in this project is the ASZED EEG Dataset (Affective State and Schizophrenia EEG Dataset).
👉 Download Dataset:
https://zenodo.org/records/14178398/files/ASZED-153.zip?download=1⁠�
📌 Dataset Details
EEG files in .edf format
Multiple brain electrodes (channels)
Clinical data including age, gender, and category
Labels:
1 → Schizophrenia Patient
0 → Healthy Control
⚠️ Note: Dataset is not included in this repository due to its large size.
After downloading, extract and place it in a local folder and update the path in the code.


⚙️ Project Workflow
EEG data loading using MNE
Signal preprocessing using bandpass filtering (0.5–40 Hz)
Feature extraction (Mean, Standard Deviation, Variance per channel)
Merging EEG features with clinical data
Data cleaning and handling missing values
Feature scaling using StandardScaler
Train-test split
Model training and evaluation
Prediction and performance analysis


🧠 Models Used
Random Forest
Support Vector Machine (SVM)
Logistic Regression
Convolutional Neural Network (CNN)


📊 Results
Random Forest → 93% Accuracy
SVM → 84% Accuracy
Logistic Regression → 79% Accuracy
CNN → 86.2% Accuracy
👉 Random Forest achieved the best performance among all models.


📈 Visualizations
The project includes the following outputs:
Confusion Matrix
ROC Curve
Model Comparison Graph
Feature Importance Graph
These visualizations help in understanding model performance and prediction behavior.


🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
TensorFlow / Keras
MNE (EEG processing)


🚀 How to Run the Project
1. Install dependencies
Bash
pip install -r requirements.txt
2. Update dataset path
Modify file paths in main.py according to your local dataset location.
3. Run the project
Bash
python main.py
4. Run Streamlit App (Optional)
Bash
streamlit run app.py


📁 Project Structure
EEG_SCHIZOPHRENIA_PROJECT/
│── assets/                  # Images / outputs
│── app.py                   # Streamlit application
│── main.py                  # Main ML pipeline
│── convert_model.py         # Model conversion script
│── requirements.txt         # Dependencies
│── runtime.txt              # Runtime configuration
│── schizophrenia_model.h5   # Trained CNN model
│── test_samples.csv         # Sample test data
│── Schizophrenia output.pdf # Project output/report
│── README.md                # Documentation


⚠️ Important Notes
Ensure dataset paths are correctly updated
CNN input shape depends on extracted feature size
Large dataset may require good system performance


🔮 Future Improvements
Use larger and more diverse EEG datasets
Apply advanced deep learning models (RNN, Transformers)
Implement real-time EEG signal processing
Improve interpretability using explainable AI techniques

👨‍💻 Author
Ch. Karthikeya


⭐ Conclusion
This project demonstrates that machine learning and deep learning models can effectively analyze EEG signals to detect schizophrenia. The hybrid approach improves classification performance and highlights the potential of AI in supporting neurological disorder diagnosis.
