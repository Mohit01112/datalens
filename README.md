# 🔍 DataLens — AI-Powered EDA & Dashboard Generator

> **Turn Data Into Insights.**

DataLens is an AI-powered exploratory data analysis (EDA) application built with Streamlit and Groq. It uses **`openai/gpt-oss-20b`** to determine what should be analysed, while Python performs the actual calculations from the uploaded dataset.

**Core principle:** AI decides what to analyse. Python calculates the results.

## 🚀 Live Demo

urlDataLens Live Demohttps://datalens-epg8zxqbkm7bytqm5j3ff7.streamlit.app/

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Smart Analysis** | Upload CSV/Excel files and let AI design relevant KPIs, charts, and insights. |
| 🔒 **No Hallucinated Numbers** | KPIs and chart values are calculated from the actual dataset using Pandas and Plotly. |
| 📊 **Interactive Charts** | Interactive Plotly visualizations with hover, zoom, and export support. |
| 📈 **10 Chart Types** | Bar, line, scatter, histogram, box, pie, heatmap, treemap, sunburst, and funnel. |
| 🔎 **Dynamic Filters** | Automatically generated filters based on dataset columns. |
| 📌 **KPI Deltas** | Calculated metric comparisons with up/down indicators where applicable. |
| 📑 **Tabbed Analysis** | Organised views for KPIs, Charts, Insights, and Code. |
| 🔄 **Auto-Retry** | Exponential backoff for transient API failures. |
| ⚡ **Analysis Caching** | Reuses analysis results for the same dataset profile and request. |
| 📅 **Smart Date Handling** | Automatically detects date columns for time-based analysis. |
| 💻 **EDA Code Export** | Generates downloadable Python EDA code. |
| 🧠 **AI Insights** | Produces business-oriented observations from the dataset profile. |

---

## 🧠 How DataLens Works

DataLens separates **AI reasoning** from **numerical computation**.

### AI handles

- Selecting useful KPIs
- Selecting appropriate charts
- Choosing relevant dataset columns
- Identifying areas that should be investigated for insights

### Python handles

- Numerical calculations
- KPI computation
- Data aggregation
- Chart generation
- Plan validation

This architecture helps prevent the AI from inventing numerical results.

---

## 🏗️ Architecture

```text
                         User
                           │
                           ▼
                  Upload CSV / Excel
                           │
                           ▼
                ┌─────────────────────┐
                │   DataLens Loader   │
                └──────────┬──────────┘
                           │
                           ▼
                  Data Profiling
                      (Pandas)
                           │
                           ▼
                ┌─────────────────────┐
                │      Groq LLM       │
                │ openai/gpt-oss-20b  │
                └──────────┬──────────┘
                           │
                           ▼
                    Analysis Plan
                           │
                           ▼
                  Python Validation
                           │
                    ┌──────┴──────┐
                    ▼             ▼
               KPI Engine    Chart Engine
                 Pandas          Plotly
                    │             │
                    └──────┬──────┘
                           ▼
                Interactive Dashboard
                           │
                           ▼
                     AI Insights
```

---

## 📁 Project Structure

```text
datalens/
│
├── app.py
│   └── Streamlit UI and application orchestration
│
├── config.py
│   └── Groq API configuration
│
├── requirements.txt
│
├── .env
│   └── Local environment variables
│
├── .env.example
│   └── Environment variable template
│
├── .gitignore
│
├── sample_sales_data.csv
│
├── utils/
│   ├── __init__.py
│   │
│   ├── data_loader.py
│   │   └── File loading, date parsing, and data profiling
│   │
│   ├── prompts.py
│   │   └── AI prompt templates
│   │
│   ├── groq_client.py
│   │   └── Groq LLM client with retry handling
│   │
│   ├── eda_engine.py
│   │   └── Plan validation, KPI calculation, and chart generation
│   │
│   └── filters.py
│       └── Dynamic sidebar filter generation
│
└── outputs/
    ├── generated_code/
    │   └── Generated EDA scripts
    │
    └── generated_images/
        └── Generated output images
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd datalens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Groq

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Only the API key and model are stored in `.env`.

**Never commit your real `.env` file to GitHub.**

### 4. Run DataLens

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔐 Environment Configuration

DataLens uses two environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

These values are loaded through `config.py`.

The model is configured through the environment rather than being hard-coded into the application, allowing it to be changed without modifying the source code.

---

## 🔒 Preventing Hallucinated Numbers

The most important design principle in DataLens is that the LLM does **not** calculate the numerical results displayed to the user.

### Workflow

```text
User Request
     │
     ▼
Build Data Profile
     │
     ▼
Send Profile to Groq
     │
     ▼
AI Returns Analysis Plan
     │
     ▼
Validate Plan
     │
     ├── Validate columns
     ├── Validate chart types
     └── Validate operations
     │
     ▼
Compute KPIs
     │
     ▼
Build Charts
     │
     ▼
Display Results
```

### Example

If the dataset contains:

```text
Sales
100
200
300
```

The LLM can decide that **Average Sales** is a useful KPI.

Python then performs the calculation:

```python
df["Sales"].mean()
```

The calculated result is what DataLens displays.

> **AI decides what to analyse. Python calculates the result.**

---

## 🔄 Resilience & Reliability

### Auto-Retry

Transient API failures are handled using exponential backoff:

```text
Attempt 1
   ↓
1 second
   ↓
Attempt 2
   ↓
2 seconds
   ↓
Attempt 3
   ↓
4 seconds
   ↓
Final attempt
```

This helps handle temporary API failures and rate limits.

### Token Budget Protection

Large dataset profiles are controlled before being sent to the LLM to reduce unnecessary token usage and context-window problems.

### Malformed JSON Handling

If the LLM returns an invalid analysis plan, DataLens handles the JSON parsing failure instead of allowing the application to crash.

### Column Validation

AI-generated plans are validated against the uploaded dataset. Suggestions referring to unavailable columns can be removed during validation.

### Analysis Caching

Analysis results can be cached using a combination of:

```text
Dataset Profile
      +
User Request
```

This allows repeated analysis requests for the same dataset and prompt to reuse cached results.

---

## 📊 Supported Visualizations

DataLens supports 10 visualization types:

1. Bar Chart
2. Line Chart
3. Scatter Plot
4. Histogram
5. Box Plot
6. Pie Chart
7. Heatmap
8. Treemap
9. Sunburst
10. Funnel Chart

The AI selects visualizations based on the structure of the dataset and the user's request.

---

## 🔎 Dynamic Data Filters

DataLens automatically generates filtering controls from the uploaded dataset.

```text
Dataset
   │
   ├── Numeric Column
   │       └── Range Slider
   │
   ├── Categorical Column
   │       └── Multi-select
   │
   ├── Date Column
   │       └── Date Filter
   │
   └── Boolean Column
           └── Checkbox
```

Applying filters updates the analysis context accordingly.

---

## 📌 KPI Generation

DataLens can identify useful KPI calculations such as:

```text
Count
Sum
Mean
Median
Minimum
Maximum
Standard Deviation
```

All KPI values are calculated from the uploaded DataFrame using Python.

---

## 💻 EDA Code Generation

DataLens can generate a standalone Python EDA script based on the selected analysis plan.

```text
User Dataset
     ↓
AI Analysis Plan
     ↓
EDA Code Generator
     ↓
Python Script
     ↓
Download
```

The generated script can be downloaded directly from the application.

---

## 🛡️ Security

The Groq API key should **never** be written directly inside Python source files.

Use environment variables:

```env
GROQ_API_KEY=your_key
GROQ_MODEL=openai/gpt-oss-20b
```

Add `.env` to `.gitignore`:

```gitignore
.env
```

**Never upload your real API key to GitHub.**

---

## ☁️ Deployment

DataLens can be deployed on platforms that support Python and Streamlit, including:

- Streamlit Community Cloud
- Docker
- Google Cloud
- Other compatible cloud platforms

For cloud deployment, configure API keys and other environment variables through the platform's secrets/environment configuration instead of committing `.env` to the repository.

---

## 🛠️ Tech Stack

### Frontend

- Streamlit

### AI / LLM

- Groq
- `openai/gpt-oss-20b`

### Data Processing

- Python
- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

### Configuration

- python-dotenv

### Development

- Git
- GitHub

---

## 🎯 Project Goal

DataLens is designed to reduce the manual effort involved in exploratory data analysis.

### Traditional workflow

```text
Load Dataset
     ↓
Understand Columns
     ↓
Find Missing Values
     ↓
Choose KPIs
     ↓
Choose Charts
     ↓
Write Visualization Code
     ↓
Analyse Results
```

### DataLens workflow

```text
Upload Dataset
     ↓
Ask a Question
     ↓
AI Understands the Dataset
     ↓
Automatic EDA
     ↓
KPIs + Charts + Insights
     ↓
Interactive Dashboard
```

DataLens combines AI-assisted analysis with deterministic Python-based computation to make exploratory data analysis faster and more reproducible.

---

## 👨‍💻 Author

**Mohit Jadhav**

AI & Data Science Engineer

---

## 📝 License

This project is licensed under the **MIT License**.
