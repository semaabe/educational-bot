# crudChroma.py
import chromadb, uuid, os, urllib.parse, asyncio, logging, time


from utlis.config import DB_PATH
from databases.chroma.modelsChroma import generate_embedding
from services.getPdfs import read_hyperlinks, match_filenames_to_urls
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import logging

logger = logging.getLogger("crudChroma")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Shared by save_pdfs (course materials) and save_pdfs_from_discord (!upload)
# so both paths produce chunks of the same size.
PDF_CHUNK_SIZE = 1000
PDF_CHUNK_OVERLAP = 200


def create_pdf_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size = PDF_CHUNK_SIZE,
        chunk_overlap = PDF_CHUNK_OVERLAP
    )


class CRUD():
    def __init__(self):
        self.client = chromadb.PersistentClient(path = DB_PATH)
        
    async def save_to_db(self, data):
        # logger.info(f"Saving {data} to db")
        for item in data:
            collection_name, document, embedding, metadata = (
                item['collection_name'], item['document'], item['embedding'], item.get('metadata', {})
            )

            # Change collection_name to type str since it was an int, but has to
            # be a str in order to be used as a collection name
            collection_name = str(collection_name)

            # Note that collection_name is equivalent to channel_id
            collection = await asyncio.to_thread(self.client.get_or_create_collection, collection_name)

            # Handle both document types: (1) LangChain Document object, (2) plain string
            if hasattr(document, 'metadata') and hasattr(document, 'page_content'):
                # LangChain Document
                doc_id = str(document.metadata.get('id', 'unknown'))
                doc_content = document.page_content
                doc_metadata = document.metadata
            else:
                # Plain string (from Discord upload)
                doc_id = str(metadata.get('id', str(uuid.uuid4())))
                doc_content = document
                doc_metadata = metadata

            await asyncio.to_thread(
                collection.upsert,
                ids=[doc_id],
                documents=[doc_content],
                embeddings=[embedding],
                metadatas=[doc_metadata]
            )

            print(f"'{doc_content[:50]}' is added to the collection {collection_name}")

    async def get_data_by_similarity(self, collection_name, query_embedding, top_k=10):
        start = time.perf_counter()
        try:
            # Generate the embedding for the query
            print(f"Retrieving documents for the collection: {collection_name}")

            # Get or create the collection if it doesn't exist
            collection = await asyncio.to_thread(self.client.get_or_create_collection, collection_name)
            print(f"Collection retrieved: {collection}")

            # Query the collection
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            logging.info(f"get_data_by_similarity {collection_name} took {(time.perf_counter() - start) * 1000:.2f}ms")
            return results

        except Exception as e:
            print(f"Error with retrieving relevant history: {e}")
            return []
    async def get_data_by_id(self, collection_name, ids):
        # convert ids to str
        ids = [str(id) for id in ids]
        start = time.perf_counter()
        try:
            collection = await asyncio.to_thread(self.client.get_or_create_collection, collection_name)
            results = await asyncio.to_thread(
                collection.get,
                ids=ids,
                # where={"style": "style1"}
            )

            logging.info(f"get_data_by_id {collection_name} took {(time.perf_counter() - start) * 1000:.2f}ms")
            return results

        except Exception as e:
            print(f"Error with retrieving data by id: {e}")
            return []

    async def delete_collection(self, name):
        start = time.perf_counter()
        try:
            await asyncio.to_thread(self.client.delete_collection, name=name)
            logging.info(f"delete_collection {name} took {(time.perf_counter() - start) * 1000:.2f}ms")
        except Exception as e:
            print(f"Error with deleting collection: {e}")
            raise

    async def get_all_documents(self, collection_name):
        start = time.perf_counter()
        try:
            collection = await asyncio.to_thread(self.client.get_or_create_collection, collection_name)
            results = await asyncio.to_thread(collection.get)
            logging.info(f"get_all_documents {collection_name} took {(time.perf_counter() - start) * 1000:.2f}ms")
            return results
        except Exception as e:
            print(f"Error with retrieving all documents: {e}")
            return []
    
    async def save_pdfs(self, file_path, collection_name):
        # Use file_path as-is (already absolute from caller)
        if not file_path:
            file_path = '/app/pdfs'
        
        print(f"Saving PDFs from {file_path} to collection {collection_name}")
        logging.info(f"save_pdfs: Looking for PDFs at {file_path}")
        
        # Check if directory exists
        if not os.path.exists(file_path):
            error_msg = f"Directory not found: {file_path}"
            print(f"crudChroma.py: Error with loading PDFs: {error_msg}")
            logging.error(f"crudChroma.py: {error_msg}")
            return []  # Return empty list instead of None
        
        # Get the files
        try:
            loader = DirectoryLoader(file_path, glob = "*.pdf", show_progress = True)
            docs = loader.load()
            logging.info(f"save_pdfs: Successfully loaded {len(docs)} PDF documents")

        except Exception as e:
            print(f"crudChroma.py: Error with loading PDFs: {e}")
            logging.error(f"crudChroma.py: Error with loading PDFs from {file_path}: {e}")
            return []  # Return empty list instead of None
        

        # split the text
        text_splitter = create_pdf_text_splitter()

        # get docs, ids, and filenames (for metadata purposes)
        docs = text_splitter.split_documents(docs)
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]

        # remove the file path and extension from the source
        filenames = [doc.metadata['source'].split('/')[-1].split('.')[0] for doc in docs]

        # get the hyperlinks for the pdfs with filenames
        hyperlinks_file = f"{file_path}/../hyperlinks.csv"
        urls = read_hyperlinks(hyperlinks_file)
        matched_urls = match_filenames_to_urls(filenames, urls)

        for filename, url in matched_urls.items():
            print(f"{filename}: {url}")

        # save the docs in the collection with collection_name
        collection = self.client.get_or_create_collection(collection_name)

        data_to_save = []
        for doc, id, filename in zip(docs, ids, filenames):
            url, text = matched_urls.get(filename, (None, "Description not found"))
            combined_text = f"{text} {doc.page_content}"
            # print the first 50 characters of the combined text
            print(f"Combined text: {combined_text[:50]}...")
            embedding = await generate_embedding(combined_text)
            
            # Format the filename
            filename = urllib.parse.unquote(filename.replace('_', ' '))
            
            if url:
                source = f"[{filename}]({url})"
            else:
                source = filename
            
            # Prepare the metadata for saving
            metadata = {
                "id": id,
                "source": source
            }

            # Add to data to be saved
            data_to_save.append({
                "collection_name": collection_name,
                "document": doc,
                "embedding": embedding,
                "metadata": metadata
            })

        return data_to_save
    
    async def save_pdfs_from_discord(self, user, collection_name, filename, text):
        """Split one uploaded PDF's text into chunks and embed each chunk.

        Returns the items to pass to save_to_db, one per chunk. Returns an
        empty list when the text has nothing to index (e.g. a scanned PDF).
        """
        print(f"Saving {filename} from {user} to collection {collection_name}")

        chunks = [
            chunk for chunk in create_pdf_text_splitter().split_text(text or "")
            if chunk.strip()
        ]

        data_to_save = []
        for chunk_index, chunk in enumerate(chunks):
            embedding = await generate_embedding(chunk)

            # Chroma metadata values must be str, int, float or bool
            metadata = {
                "id": str(uuid.uuid4()),
                "source": filename,
                "filename": filename,
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
                "uploader": str(user),
            }

            data_to_save.append({
                "collection_name": collection_name,
                "document": chunk,
                "embedding": embedding,
                "metadata": metadata
            })

        return data_to_save

    async def delete_documents(self, collection_name, ids):
        """Remove documents by id; used to roll back a partially saved upload."""
        if not ids:
            return
        collection = await asyncio.to_thread(self.client.get_or_create_collection, str(collection_name))
        await asyncio.to_thread(collection.delete, ids=[str(id) for id in ids])

    async def get_collection_info(self, name: str):
        # Get information aboput a part icualr collection 
        try:
            collection = self.client.get_collection(name)
            count = collection.count()
            metadata = collection.metadata or {}

            # Get a sample of document IDs (or other identifying info)
            docs = collection.get(limit=3)  # Adjust as needed; may need to use your DB's API
            sample_docs = docs.get("ids", []) if docs else []

            return {
                "name": collection.name,
                "description": metadata.get("description"),
                "metadata": metadata,
                "document_count": count,
                "created_at": metadata.get("created_at"),
                "sample_documents": sample_docs  # Add this line
            }
            
        except Exception as e:
            print(f"Error getting collection info for '{name}': {e}")
            return {"error": f"Collection '{name}' not found"}

    # Collection Management Methods
    async def create_collection(self, name: str, description: str = None, metadata = None):
        """Create a new collection explicitly"""
        try:
            # Check if collection already exists
            try:
                existing_collection = self.client.get_collection(name)
                if existing_collection:
                    return {"error": f"Collection '{name}' already exists"}
            except:
                # Collection doesn't exist yet, that's expected - proceed to create
                pass

            # Create the collection
            collection = self.client.create_collection(
                name=name,
                metadata={"description": description, **(metadata or {})}
            )
            
            print(f"Collection '{name}' created successfully")
            return {"message": f"Collection '{name}' created successfully", "collection": collection}
            
        except Exception as e:
            print(f"Error creating collection '{name}': {e}")
            return {"error": str(e)}
    
    async def delete_collection(self, name: str):
        """Deletes a collection"""
        try:
            try:
                collection = self.client.get_collection(name)
                if collection:
                    self.client.delete_collection(name)
                    print(f"Collection '{name}' deleted successfully")
                    return {"message": f"Collection '{name}' deleted successfully"}
            except Exception as e:
                return {"error": f"Collection '{name}' not found"}
                
        except Exception as e:
            print(f"Error deleting collection '{name}': {e}")
            return {"error": str(e)}
    
    async def list_collections(self) -> list[dict]:
        """Return all collections with normalized metadata."""
        try:
            collections = await asyncio.to_thread(
                self.client.list_collections
            )

            collection_list = []

            for collection in collections:
                metadata = collection.metadata or {}

                document_count = await asyncio.to_thread(
                    collection.count
                )

                collection_list.append({
                    "name": collection.name,
                    "description": metadata.get("description"),
                    "metadata": metadata,
                    "document_count": document_count,
                    "created_at": metadata.get("created_at"),
                })

            return collection_list

        except Exception as e:
            logging.exception("Error listing collections: %s", e)
            return []
        
