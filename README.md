# DocExtract

DocExtract is a Python tool for extracting, processing, and analyzing documents using computer vision and generative AI models. It leverages Streamlit for interactive UI, OpenCV for image processing, EasyOCR for optical character recognition, and integrates with Google Generative AI and Groq APIs.

## Features
- Document extraction and analysis
- OCR with EasyOCR
- Image processing with OpenCV
- Generative AI integration (Google, Groq)
- Streamlit web interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/doc_extract.git
   cd doc_extract
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run docextract.py
```

## Configuration

- Add your API keys and configuration in the `.env` file.

## Dependencies
- streamlit
- opencv-python
- numpy
- Pillow
- groq
- easyocr
- python-dotenv
- PyMuPDF
- google-generativeai

## License

This project is licensed under the MIT License.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or support, open an issue or contact the maintainer at yourusername@domain.com.
