from setuptools import setup, find_packages

setup(
    name="healthcare-mrag",
    version="2.0.0",
    description="Healthcare Multimodal RAG for Medical Visual Question Answering",
    author="Gokul",
    license="Apache-2.0",
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
)
