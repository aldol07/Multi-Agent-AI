from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    JSONLoader,
    TextLoader
)
from langchain.schema import Document
import json
import io
import logging
import tempfile
import os
import time
import re

class DocumentLoader:
    @staticmethod
    def load_document(file_content: bytes, file_type: str, file_name: str) -> List[Document]:
        """Load document based on file type using appropriate LangChain loader."""
        try:
            
            file_type = file_type.lower()
            
            if file_type == 'pdf':
                
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                try:
                    
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    return docs
                finally:
                    
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logging.error(f"Error deleting temporary PDF file {temp_path}: {e}")
            
            elif file_type == 'json':
                try:
                    
                    json_str = file_content.decode('utf-8')
                    json_data = json.loads(json_str)
                    
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        json.dump(json_data, temp_file)
                        temp_file.flush()
                        temp_path = temp_file.name
                    
                    try:
                       
                        loader = JSONLoader(
                            file_path=temp_path,
                            jq_schema='.',
                            text_content=False
                        )
                        docs = loader.load()
                        return docs
                    finally:
                        
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logging.error(f"Error deleting temporary JSON file {temp_path}: {e}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format: {str(e)}")
            
            elif file_type == 'txt':
                
                content = file_content.decode('utf-8')
                
                
                if '@' in content and ('Subject:' in content or 'From:' in content):
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file.flush()
                        temp_path = temp_file.name
                    
                    try:
                       
                        loader = TextLoader(temp_path, encoding='utf-8')
                        docs = loader.load()
                        
                        
                        for doc in docs:
                            doc.metadata['type'] = 'EMAIL'
                            
                            lines = doc.page_content.split('\n')
                            for line in lines:
                                if line.startswith('From:'):
                                    doc.metadata['sender'] = line[5:].strip()
                                elif line.startswith('Subject:'):
                                    doc.metadata['subject'] = line[8:].strip()
                        
                        return docs
                    finally:
                        
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logging.error(f"Error deleting temporary text file {temp_path}: {e}")
                else:
                    # Regular text file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file.flush()
                        temp_path = temp_file.name
                    
                    try:
                        # Load text using the temporary file
                        loader = TextLoader(temp_path, encoding='utf-8')
                        docs = loader.load()
                        return docs
                    finally:
                        # Ensure cleanup happens even if loading fails
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logging.error(f"Error deleting temporary text file {temp_path}: {e}")
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                content = file_content.decode('latin-1')
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    temp_path = temp_file.name
                
                try:
                    # Load text using the temporary file
                    loader = TextLoader(temp_path, encoding='latin-1')
                    docs = loader.load()
                    return docs
                finally:
                    # Ensure cleanup happens even if loading fails
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logging.error(f"Error deleting temporary text file {temp_path}: {e}")
            except Exception as e:
                raise Exception(f"Failed to decode file content: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            raise Exception(f"Error loading document: {str(e)}")
    
    @staticmethod
    def extract_metadata(doc: Document) -> Dict[str, Any]:
        """Extract metadata from document."""
        metadata = doc.metadata.copy()
        
        # Add document type
        if 'source' in metadata:
            file_name = metadata['source']
            if file_name.endswith('.pdf'):
                metadata['type'] = 'PDF'
            elif file_name.endswith('.json'):
                metadata['type'] = 'JSON'
            elif file_name.endswith('.txt'):
                # Use regex to check for email pattern more reliably
                email_pattern = re.compile(r'^From:.*?\n.*Subject:.*\n', re.DOTALL | re.IGNORECASE)
                if email_pattern.search(doc.page_content):
                    metadata['type'] = 'EMAIL'
                else:
                    metadata['type'] = 'TEXT'
        
        return metadata 