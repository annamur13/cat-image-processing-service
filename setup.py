# setup.py

from setuptools import setup, find_packages

setup(
    name="cat-image-processor",
    version="1.0.0",
    author="Murzina Anna",
    description="A package for downloading and processing cat images from TheCatAPI",
    long_description=open("README.md", encoding="utf-8").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),

    package_data={
        "cat_image_processor": ["logging_config.json"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cat-process=cat_image_processor.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "httpx>=0.23.0",
        "aiofiles>=22.0.0",
    ],
)