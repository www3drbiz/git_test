import os
import io
import re
import pandas as pd
import pytesseract
from PIL import Image
import pdfplumber
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class PDFDataExtractor:
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor with the given PDF file path
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.text_content = None
        self.images = []
        self.tables = []

    def extract_text(self) -> str:
        """
        Extract text from PDF using Langchain
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Use Langchain PDF Loader
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            # Combine text from all pages
            self.text_content = "\n".join([doc.page_content for doc in documents])
            return self.text_content
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None

    def extract_images(self, output_dir: str = 'extracted_images') -> List[str]:
        """
        Extract images from PDF using pdfplumber
        
        Args:
            output_dir (str): Directory to save extracted images
        
        Returns:
            List[str]: List of paths to extracted images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_image_paths = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    images = page.images
                    
                    for img_idx, img in enumerate(images):
                        try:
                            # Extract image
                            x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                            cropped_page = page.crop((x0, top, x1, bottom))
                            
                            # Convert to PIL Image
                            image = cropped_page.to_image()
                            
                            # Save image
                            image_path = os.path.join(output_dir, f'page_{page_num}_image_{img_idx}.png')
                            image.save(image_path)
                            
                            extracted_image_paths.append(image_path)
                            self.images.append(image_path)
                        except Exception as img_error:
                            print(f"Error extracting image: {img_error}")
            
            return extracted_image_paths
        except Exception as e:
            print(f"Error processing PDF images: {e}")
            return []

    def extract_tables(self) -> List[pd.DataFrame]:
        """
        Extract tables from PDF using pdfplumber
        
        Returns:
            List[pd.DataFrame]: List of extracted tables as pandas DataFrames
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract tables
                    tables = page.extract_tables()
                    
                    for table in tables:
                        # Convert to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])
                        self.tables.append(df)
            
            return self.tables
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return []

    def extract_tables_with_ocr(self) -> List[pd.DataFrame]:
        """
        Extract tables using OCR when traditional methods fail
        
        Returns:
            List[pd.DataFrame]: List of extracted tables as pandas DataFrames
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Convert page to image
                    img = page.to_image()
                    img_path = f'page_{page_num}_for_ocr.png'
                    img.save(img_path)
                    
                    # Use Tesseract OCR to extract table
                    ocr_table = pytesseract.image_to_data(
                        Image.open(img_path), 
                        output_type=pytesseract.Output.DATAFRAME
                    )
                    
                    # Process OCR data into a table
                    processed_table = self._process_ocr_data(ocr_table)
                    
                    if not processed_table.empty:
                        self.tables.append(processed_table)
                    
                    # Clean up temporary image
                    os.remove(img_path)
            
            return self.tables
        except Exception as e:
            print(f"Error extracting tables with OCR: {e}")
            return []

    def _process_ocr_data(self, ocr_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process OCR data into a structured table
        
        Args:
            ocr_data (pd.DataFrame): OCR output DataFrame
        
        Returns:
            pd.DataFrame: Processed table
        """
        # Implement custom logic to convert OCR data to a table
        # This might involve grouping text by rows and columns
        # Placeholder implementation - you'll need to customize based on your specific PDFs
        processed_table = pd.DataFrame()
        
        return processed_table

    def semantic_search(self, query: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Perform semantic search on PDF content
        
        Args:
            query (str): Search query
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        
        Returns:
            List[str]: Relevant text chunks
        """
        if not self.text_content:
            self.extract_text()
        
        # Use Langchain's text splitter for semantic search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split text into chunks
        texts = text_splitter.split_text(self.text_content)
        
        # Simple semantic search (you might want to use embedding models for more advanced search)
        relevant_chunks = [
            chunk for chunk in texts 
            if query.lower() in chunk.lower()
        ]
        
        return relevant_chunks

def main():
    # Example usage
    pdf_path = './[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf'
    extractor = PDFDataExtractor(pdf_path)
    
    # Extract text
    text = extractor.extract_text()
    print("Extracted Text:")
    print(text[:500])  # Print first 500 characters
    
    # Extract images
    images = extractor.extract_images()
    print(f"\nExtracted {len(images)} images")
    
    # Extract tables
    tables = extractor.extract_tables()
    print(f"\nExtracted {len(tables)} tables")
    
    # Perform semantic search
    search_results = extractor.semantic_search("important concept")
    print("\nSemantic Search Results:")
    for result in search_results:
        print(result)

if __name__ == '__main__':
    main()

# Required dependencies (install via pip):
# langchain
# pdfplumber
# pytesseract
# pandas
# pillow
# numpy