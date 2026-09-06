"""NalarData - pustaka analisis multivariat.

Seluruh fungsi di paket ini murni komputasi (DataFrame masuk, hasil keluar)
sehingga dapat diuji tanpa menjalankan antarmuka Streamlit.
"""

from nalardata import (
    assumptions,
    cca,
    clustering,
    correlation,
    descriptive,
    discriminant,
    factor_analysis,
    io_utils,
    manova,
    pca_analysis,
    preprocessing,
    regression,
)

__all__ = [
    "assumptions",
    "cca",
    "clustering",
    "correlation",
    "descriptive",
    "discriminant",
    "factor_analysis",
    "io_utils",
    "manova",
    "pca_analysis",
    "preprocessing",
    "regression",
]

__version__ = "0.1.0"
