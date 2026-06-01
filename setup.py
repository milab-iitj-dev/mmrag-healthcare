from setuptools import setup, find_packages

setup(
    name="healthcare-mrag",
    version="2.0.0",
    description="Healthcare Multimodal RAG for Medical Visual Question Answering",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Gokul",
    license="Apache-2.0",
    url="https://github.com/milab-iitj-dev/mmrag-healthcare",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.46.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "ui": [
            "gradio>=5.0.0",
        ],
        "colqwen2": [
            "colpali-engine>=0.3.0",
        ],
        "eval": [
            "nltk>=3.8.0",
            "rouge-score>=0.1.2",
            "bert-score>=0.3.13",
        ],
        "retrieval": [
            "rank-bm25>=0.2.2",
            "faiss-cpu>=1.7.4",
            "open-clip-torch>=2.24.0",
            "sentence-transformers>=2.6.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "matplotlib>=3.8.0",
        ],
    },
)
