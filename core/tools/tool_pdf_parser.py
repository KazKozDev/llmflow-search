"""
PDF Parser Tool - Extract text from PDF files.
"""
import logging
import aiohttp
import io
from typing import Optional

logger = logging.getLogger(__name__)


class PDFParserTool:
    """Tool for parsing PDF files."""
    
    def __init__(self):
        self.name = "parse_pdf"
        self.description = "Extract text content from PDF files"
        
    async def parse_pdf(self, url: str, max_chars: int = 10000) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            url: URL of the PDF file
            max_chars: Maximum characters to extract
            
        Returns:
            Extracted text content
        """
        try:
            # Import PDF library (try multiple options)
            pdf_library = None
            try:
                import PyPDF2
                pdf_library = "pypdf2"
            except ImportError:
                try:
                    import pdfplumber
                    pdf_library = "pdfplumber"
                except ImportError:
                    logger.error("No PDF parsing library found. Install PyPDF2 or pdfplumber.")
                    return ""
            
            # Download PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to download PDF: {url} (status: {response.status})")
                        return ""
                    
                    pdf_data = await response.read()
            
            # Parse PDF based on available library
            if pdf_library == "pypdf2":
                return await self._parse_with_pypdf2(pdf_data, max_chars)
            elif pdf_library == "pdfplumber":
                return await self._parse_with_pdfplumber(pdf_data, max_chars)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error parsing PDF {url}: {e}")
            return ""
    
    async def _parse_with_pypdf2(self, pdf_data: bytes, max_chars: int) -> str:
        """Parse PDF using PyPDF2."""
        import PyPDF2
        
        try:
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            total_chars = 0
            
            for page_num in range(len(pdf_reader.pages)):
                if total_chars >= max_chars:
                    break
                    
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    text_parts.append(page_text)
                    total_chars += len(page_text)
            
            full_text = "\n\n".join(text_parts)
            
            # Truncate if needed
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "... [truncated]"
            
            logger.info(f"Extracted {len(full_text)} chars from PDF using PyPDF2")
            return full_text
            
        except Exception as e:
            logger.error(f"PyPDF2 parsing failed: {e}")
            return ""
    
    async def _parse_with_pdfplumber(self, pdf_data: bytes, max_chars: int) -> str:
        """Parse PDF using pdfplumber."""
        import pdfplumber
        
        try:
            pdf_file = io.BytesIO(pdf_data)
            
            text_parts = []
            total_chars = 0
            
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    if total_chars >= max_chars:
                        break
                    
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(page_text)
                        total_chars += len(page_text)
            
            full_text = "\n\n".join(text_parts)
            
            # Truncate if needed
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "... [truncated]"
            
            logger.info(f"Extracted {len(full_text)} chars from PDF using pdfplumber")
            return full_text
            
        except Exception as e:
            logger.error(f"pdfplumber parsing failed: {e}")
            return ""


# Singleton instance
_pdf_parser = None

def get_pdf_parser() -> PDFParserTool:
    """Get PDF parser singleton."""
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = PDFParserTool()
    return _pdf_parser
