# AML 2 Project: Comparing Recommender Systems

Developed by **Sanjana Sankar, Lauren Tran, and Yoojeong Seo** for APANPS5205: Applied Machine Learning 2 at Columbia University.

## Project Overview
This project addresses the challenge of "choice overload" on streaming platforms by developing a sophisticated movie recommendation system. While traditional systems often rely solely on genres, our model uses a **hybrid content-based approach** that combines genre features with **Natural Language Processing (NLP)** of movie plot synopses to capture deeper semantic similarities, such as tone and narrative themes.

## Key Features
* **Hybrid Recommendation Engine**: Integrates TF-IDF vectorization of plot summaries with genre metadata to provide contextually relevant suggestions.
* **Performance Comparison**: Evaluates a traditional genre-based model against the hybrid NLP model, demonstrating superior "vibe" detection in the hybrid version.
* **Enhanced Filtering**: Strictly filters out sequels and remakes to ensure a diverse user experience.
* **Metric-Driven Results**: Achieved a Genre Consistency score of **0.4633** and a Diversity (Entropy) score of **2.6806**, significantly outperforming the baseline genre model.

## Technologies Used
* **Language**: Python
* **Web Framework**: Streamlit
* **Machine Learning/NLP**: Scikit-learn (TfidfVectorizer, Cosine Similarity)
* **Data Handling**: Pandas, DuckDB
* **Datasets**: IMDb (Title Basics, Ratings, Principals) and Kaggle (MPST: Movie Plot Synopses with Tags)

## Installation and Setup

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment or a Conda base environment as shown in the launch command.

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/SSankar17/AML2_Recommender_Systems.git
    cd AML2_Recommender_Systems
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas streamlit scikit-learn duckdb
    ```

## How to Run
To launch the recommendation system dashboard, navigate to the project directory and run the following command:

```bash
streamlit run NLP_app.py
```

## Methodology
1.  **Data Cleaning**: Ingested and merged approximately 1.36 GB of data from IMDb and Kaggle, handling missing values and exporting cleaned CSVs (`movies.csv`, `plotline.csv`, etc.).
2.  **Model Development**:
    * **Genre Model**: Baseline using traditional category matching.
    * **NLP Model**: Utilized TF-IDF to analyze plot synopses.
    * **Hybrid Model**: Combined both methods to ensure that if a user starts with a "Thriller," they receive suggestions that are both semantically similar and genre-consistent.
3.  **Evaluation**: Tested models for diversity and rating quality, ensuring the hybrid system provided smarter, more diverse sets of recommendations.
