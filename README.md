# us-equities-portfolio-demos

## Description
This project presents an advanced portfolio optimization strategy tailored for US equities. It integrates risk and alpha factors while accounting for linear transaction costs to achieve optimal performance. Central to this strategy is a Random Forest-based non-overlapping estimator, designed to enhance medium-frequency trading by effectively combining alpha factors. For a robust analysis of these factors, Alphalens is utilized to thoroughly evaluate their performance. Comprehensive backtesting and dynamic portfolio rebalancing are executed using Zipline, ensuring the strategy's practical applicability and reliability in real-world scenarios.
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)

## Installation
# Installation

This project relies on several libraries, including Zipline, Alphalens, and PyFolio, which have C-based dependencies. To simplify the installation process and avoid potential issues, we currently support installation via Conda only.

## Step 1: Install Conda

If you don't have Conda installed, you can download and install it from the [Anaconda website](https://www.anaconda.com/) or use the lightweight [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html). Follow the installation instructions provided on the website to set it up on your Mac.

## Step 2: Set Up the Environment

Once Conda is installed, you can create a new environment using the `environment.yml` file provided in the repository. This file contains all the necessary dependencies for the project.

1. Open your terminal.
2. Navigate to the directory containing the `environment.yml` file.
3. Run the following command to create and activate the environment:

    ```bash
    conda env create -f environment.yml
    conda activate pconst310
    ```

## Step 3: Ingest Quandl Data

Zipline requires Quandl data for backtesting. To ingest this data, you'll need a Quandl API key. Follow these steps to obtain and configure your API key:

### Get a Quandl API Key:

1. Visit the [NASDAQ Data Link website](https://data.nasdaq.com/) and sign up for an account.
2. Once registered, navigate to your account settings to find your API key.

### Configure the API Key:

1. Open your terminal.
2. Use a text editor to open your `.bash_profile` or `.bashrc` file. For example, you can use `nano`:

    ```bash
    nano ~/.bash_profile
    ```

3. Add the following line to the file, replacing `<your_key>` with your actual Quandl API key:

    ```bash
    export QUANDL_API_KEY=<your_key>
    ```

4. Save the file and exit the editor.
5. Run the following command to apply the changes:

    ```bash
    source ~/.bash_profile
    ```

### Ingest the Data:

Run the following command in your terminal to ingest the Quandl data:

```bash
zipline ingest -b quandl
```

## Usage
To run the portfolio optimization strategy, execute the `orchestrator.py` script located under the `src/alphalab` directory. This script orchestrates the entire process of data ingestion, model training, and portfolio rebalancing.

1. Open your terminal.
2. Navigate to the `src/alphalab` directory in your project.
3. Run the following command:

   ```bash
   python orchestrator.py
   ```
   
This will initiate the process, utilizing the integrated strategies and tools to optimize the portfolio based on the predefined parameters and data.

## Features
- Feature 1
- Feature 2

## TODO
Things to add as the next steps.

## License
This project is licensed under the MIT License.

## Visuals
![img.png](src/images/img.png)
![img_1.png](src/images/img_1.png)
![img_2.png](src/images/img_2.png)
![img_3.png](src/images/img_3.png)