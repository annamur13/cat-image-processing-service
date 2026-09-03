# Async Image Processor & Data Analytics CLI

A comprehensive Python command-line interface (CLI) application built for image processing, concurrent batch downloading, and big data analysis. The project is fully modularized, covered with unit tests, and packaged for production distribution.

## Key Features

- **Asynchronous Batch Downloading:** Non-blocking concurrent image downloading from external REST APIs using `asyncio` and `aiohttp`.
- **High-Performance Image Processing:** Multi-core parallel image processing using `ProcessPoolExecutor` to bypass Python's GIL. Includes custom matrix operations (Sobel filter, convolution kernels, gamma correction) using `NumPy` alongside `OpenCV`.
- **Memory-Efficient Data Pipeline:** Chunk-based processing of large-scale CSV datasets using Python generators, capable of handling files larger than available RAM.
- **Statistical Analytics & Data Viz:** Automated computation of statistical metrics (moving averages, variance, 95% confidence intervals, correlation coefficients) with visual reporting via `Matplotlib`.
- **Production-Ready Architecture:** Centralized logging with JSON configuration, fully isolated environments with `.env`, comprehensive test coverage (`pytest`), and standard packaging via `setuptools`.

---

## Tech Stack

- **Core:** Python 3.10+
- **Concurrency:** `asyncio`, `aiohttp`, `aiofiles`, `concurrent.futures`
- **Data Science & Analytics:** `pandas`, `numpy`, `matplotlib`
- **Computer Vision:** `opencv-python`
- **Testing & Tooling:** `pytest`, `unittest`, `python-dotenv`, `setuptools`

---

## Installation

The project is structured as a standard Python package. You can install it in editable mode locally:

```bash
pip install -e .
```

Alternatively, install the required dependencies manually:

```bash
pip install -r requirements.txt
```

Verify the installation and check the package version:

```bash
python -c "import cat_image_processor; print(f'Version: {cat_image_processor.__version__}')"
```

---

## Configuration

Before running the application, create a `.env` file in the root directory to store your external API credentials:

```env
API_KEY=your_thecatapi_com_key_here
API_URL=https://thecatapi.com
```

---

## Usage Guide

Once installed, the application provides a global console command: `cat-process`.

### 1. Image Processing CLI
Apply custom edge detection (Sobel operator) or filters to a local image file:

```bash
cat-process sobel input.png -o output.png
```
*Note: If no output path is provided, results are automatically saved in the `cat_images_output/` directory.*

### 2. Asynchronous API Fetching & Processing
Download and process multiple random cat images concurrently:

```bash
cat-process cat-api --limit 5
```

Filter by specific breeds:
```bash
cat-process cat-api --breeds Siamese Persian --limit 3
```

### 3. Data Analytics & Visualization
The data pipeline modules ingest structured climate/weather data, run stream-based statistical analysis, and generate analytical charts.

```bash
# Example execution for data pipeline module
python -m cat_image_processor.analytics --input weather_data.csv
```

---

## Production Logs & Monitoring

The application implements a robust logging system configured via `logging_config.json`. It splits log channels for optimal performance and debugging:
- **Console Output:** Clean, real-time operational messages (Level: `INFO`).
- **Log File (`app.log`):** Detailed execution flow, execution times measured via custom decorators, and low-level diagnostic data (Level: `DEBUG`).

---

## Testing

The project maintains a strict test suite covering core application logic, API mock behaviors, context managers, and custom image manipulation algorithms.

To run the full test suite with verbose output:

```bash
python -m pytest tests/ -v
```
