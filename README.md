# JMT Predictive Asset Health Dashboard

This Streamlit application demonstrates how predictive analytics transforms bridge maintenance for a Department of Transportation. It guides users through the full journey from raw data ingestion to prioritized risk-based actions.

## Project Structure

```
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw_bridge_data.csv      # Simulated source system export
│   ├── clean_bridge_data.csv    # Transformed, feature-engineered data
│   └── prediction_results.csv   # Model output and recommendations
└── README.md
```

## Getting Started

1. **Create and activate a virtual environment (recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app.**
   ```bash
   streamlit run app.py
   ```

4. **Navigate the experience.**
   - Start at the landing page to understand the scenario.
   - Move through the three chapters using the sidebar navigation or on-page Next/Previous buttons.
   - Use filters and export tools in Chapter 3 to interact with the simulated bridge portfolio.

## Deployment

Deploy on Streamlit Community Cloud or the hosting platform of your choice by pointing to `app.py` as the entry point and ensuring the `data/` directory is included in the project root.

## Data Generation

All CSV files in the `data/` directory are simulated for demonstration purposes and can be regenerated with the provided scripts or by adapting the logic in `app.py`.

## License

This project is provided for educational and demonstration purposes.
