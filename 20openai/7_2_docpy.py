import os
import warnings
warnings.filterwarnings('ignore')

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 공식문서의 내용 https://github.com/docling-project/docling
# source = 'https://arxiv.org/pdf/2408.09869'
# converter = DocumentConverter()
# result = converter.convert(source)
# print(result.document.export_to_markdown())

# Docling 변환기
converter = DocumentConverter()
# pdf ---> Docling Documnet 변환
file_path = r'C:\00AI\LLM\20openai\doc_table_sample_20251202.pdf'
result = converter.convert()
# markdown 추출 (표 구조 보존)
markdown_content = result.document.export_to_markdown()
print(markdown_content)

#