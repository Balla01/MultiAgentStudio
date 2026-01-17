from typing import List

class RecursiveCharacterTextSplitter:
    """
    A simple implementation of recursive character text splitting.
    Matches the logic in vector_retrieval/dump_data_dbfile.py
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = ["\n\n", "\n", " ", ""]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        if self._length_function(text) <= self.chunk_size:
            return [text]
        
        # Try separators
        separator = self.separators[-1]
        for sep in self.separators:
            if sep in text:
                separator = sep
                break
        
        splits = text.split(separator) if separator else list(text)
        good_splits = []
        
        for split in splits:
            if self._length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    self._merge_splits(good_splits, separator, final_chunks)
                    good_splits = []
                final_chunks.extend(self.split_text(split))
        
        if good_splits:
            self._merge_splits(good_splits, separator, final_chunks)
            
        return final_chunks

    def _length_function(self, text: str) -> int:
        return len(text)

    def _merge_splits(self, splits: List[str], separator: str, final_chunks: List[str]):
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = self._length_function(split)
            if current_length + split_len + (len(separator) if current_length > 0 else 0) > self.chunk_size:
                if current_chunk:
                    doc = separator.join(current_chunk)
                    final_chunks.append(doc)
                    
                    # Handle overlap
                    while current_length > self.chunk_overlap:
                        current_length -= self._length_function(current_chunk[0]) + len(separator)
                        current_chunk.pop(0)
                        
            current_chunk.append(split)
            current_length += split_len + (len(separator) if current_length > 0 else 0)
            
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        """
        return self.splitter.split_text(text)
