from setuptools import setup, find_packages

setup(
    name="healthcare-mrag",
    version="0.2.0",
    description="Healthcare Multimodal RAG — Medical Visual Question Answering with Retrieval-Augmented Generation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
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
