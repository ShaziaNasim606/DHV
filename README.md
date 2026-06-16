# 📊 Data Handling & Visualisation — Global Economic Indicators

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20Matplotlib%20%7C%20Seaborn-green)
![Data](https://img.shields.io/badge/Data-World%20Bank%20Open%20Data-orange)

## 📌 Project Overview

This project performs **data handling, cleaning, and multi-format visualisation** 
of global economic indicators across **6 countries** from **2012 to 2022**, 
using real-world World Bank datasets.

It demonstrates a complete data pipeline: ingestion → cleaning → transformation 
→ visualisation → composite dashboard output.

**Countries Analysed:**
Canada, China, Germany, United Kingdom, India, United States

---

## 🗂️ Repository Structure

| File | Description |
|---|---|
| `22091562.py` | Main Python script — full data pipeline and visualisation |
| `GDP Growth.csv` | World Bank GDP Growth dataset |
| `inflation.csv` | World Bank Inflation dataset |
| `Fuel imports.csv` | World Bank Fuel Imports dataset |
| `Trade.csv` | World Bank Trade dataset |
| `urban population.csv` | World Bank Urban Population dataset |
| `rural population.csv` | World Bank Rural Population dataset |

---

## 🛠️ Tools & Technologies

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Data Source:** World Bank Open Data
- **Environment:** Spyder / Jupyter

---

## 🔍 Data Pipeline

### 1️⃣ Data Ingestion & Cleaning (`read_data`)
- Loaded 6 World Bank CSV files simultaneously
- Filtered for 6 selected countries across years 2012–2022
- Handled missing values using **row-wise mean imputation**
- Transposed data for time-series visualisation (countries as columns, years as rows)

### 2️⃣ Visualisations Generated

| Chart Type | Function | Description |
|---|---|---|
| **Line Plot (All Countries)** | `create_line_plot` | Trends over 2012–2022 with % change labels |
| **Line Plot (Per Country)** | `create_line_plot_for_country` | Individual country trend for each indicator |
| **Bar Chart** | `create_bar_chart` | Year-by-year comparison across countries |
| **2022 Overview Bar Chart** | `create_2022_bar_chart` | Multi-panel subplot of all indicators for 2022 |
| **Pie Chart (Mean)** | `create_pie_chart` | Country share of mean indicator value 2012–2022 |
| **2022 Pie Chart** | `create_2022_pie_chart` | Country share of each indicator in 2022 |
| **Composite Dashboard** | Final figure | 6-panel composite image saved as `final_image_graphs.png` |

### 3️⃣ Final Dashboard Output
- Combines 6 individual plots into a single composite figure
- Includes: Inflation Line Plot, GDP Growth Bar Chart, Fuel Imports Pie Chart,
  Trade Line Plot, Urban Population Pie Chart, 2022 Data Overview

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/ShaziaNasim606/DHV.git
cd DHV

# Install dependencies
pip install pandas numpy matplotlib seaborn

# Run the script
python 22091562.py
```

> All 6 CSV data files must be in the same directory as `22091562.py`.  
> Output charts are saved automatically as `.png` files in the same directory.

---

## 📈 Key Outputs

- `final_image_graphs.png` — composite 6-panel economic dashboard
- `Inflation_lineplot.png` — inflation trends across countries
- `GDP_Growth_lineplot.png` — GDP growth comparison
- `inflation_PieChart.png` — country share of inflation 2012–2022
- `2022 Data Overview.png` — all indicators for 2022 in one chart

---

## 💡 Key Skills Demonstrated

- End-to-end data pipeline: ingestion, cleaning, transformation, visualisation
- Reusable, well-documented Python functions with docstrings
- Multi-format visualisation: line, bar, pie, and composite dashboard charts
- World Bank data handling and country-level filtering
- Automated chart saving for reporting and presentations

---

## 👩‍💻 Author

**Shazia Nasim**  
MSc Data Science | University of Hertfordshire  
📍 Bristol, UK | 📧 nasim.shazia1@gmail.com  
🔗 [GitHub Profile](https://github.com/ShaziaNasim606)
