# LangchainRAG

LangchainRAG is an interactive application that leverages a large language model enhanced with a web search tool to answer questions related to Langchain. The application can be run in an interactive command-line mode or as a web application using Streamlit.

## Features
- Interactive command-line interface
- Web application interface using Streamlit
- Utilizes a language model with web search capabilities

## Setup Instructions

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/LangchainRAG.git
    cd LangchainRAG
    ```

2. **Install Poetry:**
    ```sh
    pip install poetry
    ```

3. **Install the required dependencies:**
    ```sh
    poetry install
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory.
    - Add your environment variables in JSON format under the key `ENV_VARIABLES`.

    Example `.env` file:
    ```env
    ENV_VARIABLES='{"OPENAI_API_KEY": "your_api_key_here"}'
    ```

## Usage

### Command-Line Interface

To run the application in interactive command-line mode:
```sh
poetry run python src/langchain_rag/main.py
```
Type your questions and get responses from the agent. Type `q`, `quit`, or `exit` to terminate the session.

### Web Application

To run the application as a web interface using Streamlit:
```sh
poetry run streamlit run src/langchain_rag/app.py
```
Open your web browser and navigate to the URL provided by Streamlit to interact with the application.

## License
This project is licensed under the MIT License.