# Synthetic Data Generation for Benchmark for Managers Data

![Synthetic Data](https://example.com/synthetic_data_image.png)

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Generating Synthetic Data](#generating-synthetic-data)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Benchmark for Managers dataset contains sensitive and confidential information that cannot be directly shared or used for certain purposes. Synthetic data generation addresses these privacy concerns by creating artificial data that closely mimics the statistical characteristics of the original dataset while not disclosing any real individual information. Furthermore, the data can be generated for other projects such as reporting, machine learning modeling, etc. This repository offers a synthetic data generation solution specifically tailored for the Benchmark for Managers dataset.

## Getting Started

### Prerequisites

Before you can generate synthetic data, make sure you have the following:

- Python 3.11+
- Virtual environment (recommended)

Conda is recommended for virtual environment which can be downloaded from https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html.



### Installation

1. Clone this repository to your local machine:

All required packages are listed in conda.yml. To use the environment, use the command below. You can choose any name for virtual environment.

```shell

conda create -n vsynth -f conda.yml
conda activate vsynth

```

If you prefer to use pip, you can use the command below that uses the requirements.txt file, which contains pip packages for the project

```shell
 pip install -r requirements.txt
```


## Usage

## Roadmap