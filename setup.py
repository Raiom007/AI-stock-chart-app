from setuptools import setup, find_packages

setup(
    name="ai_stock_chart",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "pydantic",
        "yfinance",
        "alpha_vantage",
        "pandas",
        "numpy",
        "ta",
        "scikit-learn",
        "tensorflow",
        "opencv-python"
    ],
)
