import setuptools

setuptools.setup(
    name="GeneralRLHF",
    version="0.0.1",
    author="Kayendns, gregorfr",
    author_email="",
    description="This projects aims to provide a framework for doing Reinforcement Learning from Human Feedback (RLHF).",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kayendns/RLArena",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "pymongo",
        "gymnasium",
        "opencv-python-headless",
        "PyYAML",
        "flask",
        "flask_cors",  
    ],
)