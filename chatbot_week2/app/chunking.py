from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_pdf_texts(uploaded_files, extract_text_fn):
    """
    Given a list of uploaded PDF files and a text extraction function,
    extract text from each file, split into chunks, and return chunks and metadata.
    
    Args:
        uploaded_files (list): List of uploaded file objects.
        extract_text_fn (callable): Function to extract text from a single PDF file.
        
    Returns:
        all_chunks (list of str): Text chunks from all files.
        all_metadatas (list of dict): Corresponding metadata for each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    all_chunks = []
    all_metadatas = []
    for file in uploaded_files:
        text = extract_text_fn(file)
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": file.name}] * len(chunks))
    return all_chunks, all_metadatas
