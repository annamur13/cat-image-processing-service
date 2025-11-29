from setuptools import setup, find_packages

setup(
    name="cat-image-processor",
    version="1.0.0",
    author="Murzina Anna",
    description="A package for downloading and processing cat images from TheCatAPI",
    packages=find_packages(),
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
    include_package_data=True,
)