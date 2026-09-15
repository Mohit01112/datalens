# 🔍 DataLens — AI-Powered EDA & Dashboard Generator

An intelligent exploratory data analysis tool that uses **Groq's `openai/gpt-oss-20b`** to decide *what* to analyse, while computing every number from the real data in Python — **zero hallucinated numbers**.

---

## ✨ Features

| Feature                        | Description                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------- |
| **🤖 Smart Analysis**          | Upload CSV/Excel → AI designs KPIs, charts & insights                           |
| **🔒 No Hallucinated Numbers** | Every KPI and chart is computed from real data using pandas/Plotly              |
| **📊 Interactive Charts**      | Dark-themed Plotly charts with zoom, hover, and export                          |
| **📈 10 Chart Types**          | Bar, line, scatter, histogram, box, pie, heatmap, treemap, sunburst, funnel     |
| **🔎 Dynamic Filters**         | Sidebar filter widgets are automatically generated from your data               |
| **📌 KPI Deltas**              | Up/down indicators comparing calculated metrics                                 |
| **📑 Tabbed Layout**           | Clean organisation: KPIs → Charts → Insights → Code                             |
| **🔄 Auto-Retry**              | Exponential backoff on transient API errors                                     |
| **⚡ Analysis Caching**         | Same prompt on the same dataset returns cached results without another API call |
| **📅 Smart Date Handling**     | Automatically detects date columns and handles time-based visualizations        |
| **💻 EDA Code Export**         | One-click Python script generation using pandas and matplotlib                  |
| **🧠 AI Insights**             | Generates meaningful observations based on the dataset profile                  |

---

## 🧠 How DataLens Works

DataLens separates **AI reasoning** from **numerical computation**.

The LLM decides:

* Which KPIs are useful
* Which charts should be generated
* Which columns should be used
* What insights should be investigated

Python performs:

* Actual calculations
* KPI computation
* Data aggregation
* Chart generation
* Validation

This design prevents the LLM from inventing numerical results.

---

## 🏗️ Architecture

```text
                 User
                   │
                   ▼
          Upload CSV / Excel
                   │
                   ▼
          DataLens Data Loader
                   │
                   ▼
            Data Profiling
             (Pandas)
                   │
                   ▼
        ┌─────────────────────┐
        │   Groq LLM          │
        │ openai/gpt-oss-20b  │
        └──────────┬──────────┘
                   │
              Analysis Plan
                   │
                   ▼
          Python Validation
                   │
          ┌────────┴────────┐
          ▼                 ▼
      KPI Engine       Chart Engine
       Pandas             Plotly
          │                 │
          └────────┬────────┘
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
│   └── Streamlit UI + application orchestration
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
├── utils/
│   ├── __init__.py
│   │
│   ├── data_loader.py
│   │   └── File loading, date parsing and data profiling
│   │
│   ├── prompts.py
│   │   └── AI prompt templates
│   │
│   ├── grok_client.py
│   │   └── Groq LLM client with retry handling
│   │
│   ├── eda_engine.py
│   │   └── Validation, KPI calculation and chart generation
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

## 🚀 Local Setup

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

**Only the API key and model are stored in `.env`.**

Never commit your `.env` file to GitHub.

### 4. Run DataLens

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 🔐 Environment Configuration

DataLens requires only two environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The application loads these values through `config.py`.

The model is **not hard-coded** in the application.

This allows the LLM to be changed later without modifying the source code.

---

## 🔒 How "No Hallucinated Numbers" Works

```text
User Request
     │
     ▼
Build Data Profile
     │
     │  Pandas
     ▼
Send Profile to Groq
     │
     │
     ▼
AI Returns Analysis Plan
     │
     │  JSON
     ▼
Validate Plan
     │
     ├── Remove invalid columns
     ├── Validate chart types
     └── Validate requested operations
     │
     ▼
Compute KPIs
     │
     │  Pandas
     ▼
Build Charts
     │
     │  Plotly
     ▼
Display Results
```

### Key principle

**The AI does not calculate the numbers displayed to the user.**

For example, if the dataset contains:

```text
Sales
100
200
300
```

The LLM may decide:

```text
KPI:
Average Sales
```

But Python calculates:

```python
df["Sales"].mean()
```

The resulting value is then displayed by DataLens.

Therefore:

> **AI decides what to analyse. Python calculates the result.**

This significantly reduces the risk of hallucinated numerical results.

---

## 🔄 Resilience Features

### Auto-Retry

Transient API failures are handled using exponential backoff.

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

---

### Token Budget Protection

Large dataset profiles are automatically controlled before being sent to the LLM to reduce unnecessary token usage and avoid context-window problems.

---

### Malformed JSON Handling

If the LLM returns an invalid analysis plan, DataLens catches the JSON parsing error and asks the user to retry rather than crashing the application.

---

### Column Validation

If the AI suggests a column that does not exist in the dataset, DataLens validates the plan and removes the invalid suggestion.

---

### Analysis Caching

DataLens caches analysis results using a hash generated from:

```text
Dataset Profile
      +
User Request
```

If the user submits the same request for the same dataset again, the cached result can be reused without making another LLM request.

---

## 📊 Supported Visualizations

DataLens can generate multiple visualization types depending on the dataset and user request.

### Supported chart types

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

The AI selects appropriate visualizations based on the dataset structure and the user's request.

---

## 🔎 Dynamic Data Filters

DataLens automatically generates filtering controls based on the uploaded dataset.

For example:

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

Filtering the dataset updates the analysis context accordingly.

---

## 📌 KPI Generation

DataLens can automatically identify useful KPI calculations such as:

```text
Count
Sum
Mean
Median
Minimum
Maximum
Standard Deviation
```

The actual KPI values are always calculated from the DataFrame using Python.

---

## 💻 EDA Code Generation

DataLens can generate a standalone Python EDA script based on the selected analysis plan.

Example:

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

Use:

```env
GROQ_API_KEY=your_key
GROQ_MODEL=openai/gpt-oss-20b
```

And add `.env` to `.gitignore`:

```gitignore
.env
```

Never upload your real API key to GitHub.

---

## ☁️ Deployment

DataLens can be deployed using platforms such as:

* Streamlit Community Cloud
* Docker
* Google Cloud
* Other cloud platforms supporting Python/Streamlit applications

For cloud deployment, configure the required environment variables/secrets through the deployment platform rather than committing `.env` to the repository.

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### AI / LLM

* Groq
* `openai/gpt-oss-20b`

### Data Processing

* Python
* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib

### Configuration

* python-dotenv

### Development

* Git
* GitHub

---

## 🎯 Project Goal

DataLens is designed to reduce the manual work involved in exploratory data analysis.

Instead of manually performing:

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

DataLens automates the workflow:

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

---

## 👨‍💻 Author

**Mohit Jadhav**

AI & Data Science Engineer

---

## 📝 License

This project is licensed under the **MIT License**.
#
