# app.py

import httpx, uvicorn, chromadb, time, asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Depends
from typing import Union, List, Any
from datetime import datetime, timezone
from utlis.pdf_helpers import read_pdf_text
from databases.chroma.modelsChroma import generate_embedding
from router.utterances import UTTERANCES, load_persisted_utterances

import sys
import os
import logging
import subprocess

# from router.semanticRouter import process_query
from router.semanticRouter import create_router
from router.utterances import create_utterances, UTTERANCES
from backend.modelsPydantic import (
    QueryResponse, QueryRequest, UpdateChannelInfo, UpdateChatHistory, 
    UpdateGuildInfo, UpdateMemberInfo, UpdateChannelList, CollectionCreate,
    UserCreate, UserResponse
)
from services.queryLangchain import fetchGptResponse
from services.nlpTools import TextProcessor
from databases.chroma.crudChroma import CRUD
from databases.chroma.modelsChroma import (
    generate_embedding, ChatHistory, GuildInfo, ChannelInfo, MemberInfoChannel, ChannelList
)
from databases.postgres.crudPostgres import PostgresCRUD
from utlis.prompts import PROMPTS
from backend.auth import (
    require_roles, AuthenticatedUser, ADMIN_ROLES, CONTENT_ROLES, QUERY_ROLES
)
from backend.middleware.role_middleware import RoleMiddleware


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

APP_VERSION = os.getenv("APP_VERSION") or "unknown"
try:
    BUILD_HASH = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except (OSError, subprocess.CalledProcessError):
    BUILD_HASH = "dev"

app = FastAPI()
crud = CRUD()
postgres_crud = PostgresCRUD()
semantic_router = create_router(crud)
# Default-deny: every request outside PUBLIC_PATHS must carry a valid token
# with a known role. Per-route role policies are the require_roles dependencies.
app.add_middleware(RoleMiddleware)

def get_db():
    db = postgres_crud.get_connection()
    try:
        yield db
    finally:
        postgres_crud.return_connection(db)


def _ping_postgres() -> None:
    """Run the synchronous PostgreSQL health check outside the event loop."""
    connection = None
    try:
        connection = postgres_crud.get_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    finally:
        if connection is not None:
            postgres_crud.return_connection(connection)


async def _run_health_check(check, name: str) -> str:
    try:
        await asyncio.wait_for(asyncio.to_thread(check), timeout=2.0)
        return "ok"
    except asyncio.TimeoutError:
        logging.error("%s health check timed out", name)
        return "unavailable"
    except Exception:
        logging.exception("%s health check failed", name)
        return "unavailable"


@app.on_event("startup")
async def startup_event():
    try:
        # Load persisted utterances from file into the UTTERANCES map
        await load_persisted_utterances()
        
        # Pre-load persisted utterances into the router at startup
        routes = await UTTERANCES.snapshot()
        if routes:
            logging.info("Scheduling background preload of routes from UTTERANCES on startup")
            # Run heavy init in a background thread so startup doesn't block
            def _bg_setup():
                try:
                    semantic_router._setup_routes(routes)
                except Exception as e:
                    logging.error(f"Background route setup failed: {e}")

            asyncio.get_event_loop().create_task(asyncio.to_thread(_bg_setup))
    except Exception as e:
        logging.error(f"Failed to preload routes at startup: {e}")

@app.get("/health")
async def health():
    started = time.perf_counter()
    checks = {
        "chromadb": await _run_health_check(
            crud.client.list_collections, "ChromaDB"
        ),
        "postgres": await _run_health_check(_ping_postgres, "PostgreSQL"),
    }
    status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
        "build_hash": BUILD_HASH,
        "checks": checks,
        "response_time_ms": round((time.perf_counter() - started) * 1000, 2),
    }

@app.post('/channel_query', dependencies=[Depends(require_roles(*QUERY_ROLES))]) #, response_model=QueryResponse
async def channel_query(request: QueryRequest):
    try:
        query_embedding = await generate_embedding(request.query)
        collection_name = f"chat_history_{request.channel_id}"
        relevant_docs = await crud.get_data_by_similarity(collection_name, query_embedding, top_k=5)
        channel_info = await crud.get_data_by_id(f"channel_info_{request.guild_id}", [request.channel_id])

        docs = relevant_docs.get('documents', [])
        content = docs[0] if isinstance(docs, list) and len(docs) > 0 else []
        if isinstance(content, list):
            content = content[:3]

        metadata_entries = channel_info.get('metadatas', []) if isinstance(channel_info, dict) else []
        data = metadata_entries[0] if metadata_entries else {}

        logging.info(f"Relevant messages: {content}")
        logging.info(f"Channel info: {data}")

        # combine the relevant messages and channel info
        combined_data = {
            'relevant_messages': content,
            'channel_info': data
        }

        answer = await fetchGptResponse(request.query, PROMPTS['channel_summarizer'], combined_data)
        logging.info(f"Answer: {answer}")
        return {'answer': answer}

    except Exception as e:
        logging.error(f"Error with channel related question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/setup_routes', dependencies=[Depends(require_roles(*ADMIN_ROLES))]) # sets the routes in the router
async def setup_routes(routes: dict = Body(...)):
    try:
        logging.info("Attempting to setup routes\n")
        response = semantic_router._setup_routes(routes)

        if response is None:
            raise ValueError("Could not setup routes - endpoint call")
        return response
    except Exception as e:
        logging.error(f"Error with setting up routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/resource_query', dependencies=[Depends(require_roles(*QUERY_ROLES))]) #, response_model=QueryResponse
async def resource_query(request: QueryRequest):
    try:
        # response = await process_query(crud, request)
        logging.info("Attempting to do resource query")
        response = await semantic_router.process_query(request)
        # handle router errors returned as dicts
        if isinstance(response, dict) and response.get('error'):
            logging.error(f"Router error: {response.get('error')}")
            # If the index isn't ready, return 503 to indicate temporary unavailability
            if 'Index is not ready' in str(response.get('error')):
                raise HTTPException(status_code=503, detail=response.get('error'))
            raise HTTPException(status_code=500, detail=response.get('error'))

        # should generate a response based on this

        if response is None:
            raise ValueError("Process_query returned none")
    
        # Limit the documents sent to the LLM by doing a similarity search
        query = request.query
        query_embedding = await generate_embedding(query)
        similar = await crud.get_data_by_similarity(response.name, query_embedding, top_k=3)

        # Chroma returns documents as a list-of-lists for batched queries; extract the first list
        relevant_documents = []
        try:
            docs = similar.get('documents', [])
            if isinstance(docs, list) and len(docs) > 0:
                relevant_documents = docs[0]
        except Exception as e:
            relevant_documents = []
            logging.info(f'Unable to pull relevant documents: {e}')

        logging.info(f"Relevant documents (top_k=3): {relevant_documents}")

        # combine the relevant messages and channel info
        combined_data = {
            "relevant_documents": relevant_documents,
            "query": query,
            "question_type": getattr(response, 'name', str(response))
        }

        answer = await fetchGptResponse(
                    query, PROMPTS['expert_instruction'], combined_data
                )        
        logging.info(f"Answer: {answer}")
        return {'answer': answer}

    except Exception as e:
        logging.error(f"Error with course material related question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post('/update_chat_history', dependencies=[Depends(require_roles(*QUERY_ROLES))])
async def update_chat_history(request: UpdateChatHistory):
    all_messages = request.all_messages
    total_messages = sum([len(messages) for messages in all_messages.values()])
    chat_history = []
    for _, channel_messages in all_messages.items():
        for message in channel_messages:
            message_info = {
                "channel_id": message.channel_id,
                "channel_name": message.channel_name,
                "message_id": message.message_id,
                "author": message.author,
                "author_id": message.author_id,
                "content": message.content,
                "timestamp": message.timestamp,
                "profanity_score": message.profanity_score
            }
            # Pass the chat history to modelsChroma to get document and embedding
            try:
                chat_info = ChatHistory(message_info)
                document, embedding = await chat_info.to_document()
            except Exception as e:
                logging.error(f"Error with updating chat history: {e}")
            
            chat_history.append({
                "collection_name": f"chat_history_{message_info.get('channel_id')}",
                "document": document,
                "embedding": embedding
            })

    # Save chat history to Chromadb
    try:
        await crud.save_to_db(chat_history)
    except Exception as e:
        logging.error(f"Error with saving chat history: {e}")

    logging.info(f"Update complete, {total_messages} messages from {len(all_messages)} channels are loaded to the database.")
    return {"status": "Update complete"}

@app.post('/update_info', dependencies=[Depends(require_roles(*QUERY_ROLES))])
async def update_info(request: Union[UpdateGuildInfo, UpdateChannelInfo, UpdateMemberInfo, UpdateChannelList]):
    try:
        if isinstance(request, UpdateGuildInfo):
            collection_name = "guild_info"
            info = GuildInfo(request.model_dump())

        elif isinstance(request, UpdateChannelInfo):
            collection_name = f"channel_info_{request.guild_id}"
            info = ChannelInfo(request.model_dump())
        
        elif isinstance(request, UpdateMemberInfo):
            collection_name = f"member_info_{request.channel_id}"
            info = MemberInfoChannel(request.model_dump())
        
        elif isinstance(request, UpdateChannelList):
            collection_name = f"channel_list_{request.guild_id}"
            info = ChannelList(request.model_dump())
        
        document, embedding = await info.to_document()
        data = {
            "collection_name": collection_name,
            "document": document,
            "embedding": embedding
        }
        await crud.save_to_db([data])

    except Exception as e:
        logging.error(f"Error with updating guild info: {e}")

    logging.info(f"Info updated for {collection_name}")
    return {"status": "Update complete"}

@app.post('/collections', dependencies=[Depends(require_roles(*CONTENT_ROLES))])
async def create_collection(payload: CollectionCreate):
    name = payload.name
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name'")
    if not all(ch.isalnum() or ch in "_-" for ch in name):
        raise HTTPException(
            status_code=400,
            detail="Invalid collection name. Only letters, numbers, underscores, and hyphens are allowed.",
        )
    description = payload.description if payload.description is not None else ""
    metadata = payload.metadata if payload.metadata is not None else {}
    
    try:
        result = await crud.create_collection(
            name=name,
            description=description,
            metadata=metadata,
        )
        # Check for error dict immediately before proceeding
        if isinstance(result, dict) and result.get("error"):
            logging.error(f"Failed to create collection '{name}': {result.get('error')}")
            raise HTTPException(status_code=400, detail=result["error"])
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to create collection '{name}': {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create collection: {e}")

    # generate utterances based on the description
    query = f"Collection name: {name}\nCategory description: {description}"
    try:
        logging.info("attempting to create utterances")
        await create_utterances(query)
    except Exception as e:
        logging.error(f"Failed to generate utterances for collection '{name}': {e}")
    
    new_utterances = await UTTERANCES.get(name)

    if new_utterances:
        semantic_router.add_route(name=name, utterances=new_utterances)
    else:
        logging.warning(f"No utterances found for collection '{name}', route not added.")

    info = await crud.get_collection_info(name)
    if isinstance(info, dict) and info.get("error"):
        raise HTTPException(status_code=404, detail=info["error"])
    return info

@app.get('/collections', dependencies=[Depends(require_roles(*QUERY_ROLES))])
async def list_collections():
    result = await crud.list_collections()
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get('/collections/{name}', dependencies=[Depends(require_roles(*QUERY_ROLES))])
async def get_collection(name: str):
    info = await crud.get_collection_info(name)
    if isinstance(info, dict) and info.get("error"):
        raise HTTPException(status_code=404, detail=info["error"])
    return info

@app.delete('/collections/{name}', dependencies=[Depends(require_roles(*CONTENT_ROLES))])
async def delete_collection(name: str):
    result = await crud.delete_collection(name)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post('/load_course_materials', dependencies=[Depends(require_roles(*CONTENT_ROLES))])
async def load_course_materials():
    file_path = os.getenv('PDF_OUTPUT_DIR', '/app/pdfs')
    logging.info(f"load_course_materials: Using PDF_OUTPUT_DIR={file_path}")
    collection_name = "course_materials"
    try:
        data = await crud.save_pdfs(file_path, collection_name) 

        # Check if data is None or empty
        if not data:
            logging.warning(f"No PDF data returned from save_pdfs. Directory may be empty or not exist.")
            return {"message": "No PDFs found to load.", "status": "warning"}

        # save the data to the database in chunks of ten documents
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            await crud.save_to_db(data[i:i+chunk_size])

        logging.info(f"Successfully loaded {len(data)} PDF chunks into Chroma.")
        return {"message": f"PDFs loaded successfully. {len(data)} chunks saved."}
    
    except Exception as e:
        logging.error(f"app.py: Error with loading PDFs: {e}")
        return {"message": f"Failed to load PDFs: {str(e)}", "status": "error"}, 500

def _user_response(user):
    if not user or not user.get("success") or not user.get("data"):
        return None
    return user["data"]

def _validate_discord_user(user_data: UserCreate):
    if user_data.role != "student":
        raise HTTPException(status_code=403, detail="Discord users must have the student role")

@app.post("/api/users", response_model=UserResponse, dependencies=[Depends(require_roles(*ADMIN_ROLES))])
async def create_discord_user(user_data: UserCreate, db: Any = Depends(get_db)):
    _validate_discord_user(user_data)
    existing = _user_response(postgres_crud.get_user(db=db, username=user_data.username))
    if existing:
        return existing
    created = postgres_crud.create_user(
        db=db,
        username=user_data.username,
        email=user_data.email,
        role="student",
        default_collection=user_data.default_collection,
    )
    if not created.get("success"):
        raise HTTPException(status_code=400, detail=created.get("error", "Failed to create user"))
    user = _user_response(postgres_crud.get_user(db=db, user_id=created["data"]))
    if user is None:
        raise HTTPException(status_code=500, detail="User was created but could not be retrieved")
    return user

@app.get("/api/users/{username}", response_model=UserResponse, dependencies=[Depends(require_roles(*ADMIN_ROLES))])
async def get_discord_user(username: str, db: Any = Depends(get_db)):
    user = _user_response(postgres_crud.get_user(db=db, username=username))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/api/users/batch", response_model=List[UserResponse], dependencies=[Depends(require_roles(*ADMIN_ROLES))])
async def create_users_batch(users_data: List[UserCreate], db: Any = Depends(get_db)):
    processed = []
    for user_data in users_data:
        _validate_discord_user(user_data)
        existing = _user_response(postgres_crud.get_user(db=db, username=user_data.username))
        if existing:
            processed.append(existing)
            continue
        created = postgres_crud.create_user(
            db=db,
            username=user_data.username,
            email=user_data.email,
            role="student",
            default_collection=user_data.default_collection,
        )
        if not created.get("success"):
            raise HTTPException(status_code=400, detail=created.get("error", "Failed to create user"))
        user = _user_response(postgres_crud.get_user(db=db, user_id=created["data"]))
        if user is None:
            raise HTTPException(status_code=500, detail="User was created but could not be retrieved")
        processed.append(user)
    return processed
    
# @app.post('/upload_materials')
# async def upload_course_materials(collection_name: str):
#     try:
#         collection = await crud.create_collection(collection_name)
#         # data = await crud.save_pdfs(file_path, collection_name)

#         data =  

#         # save the data to the database in chunks of ten documents
#         chunk_size = 10
#         for i in range(0, len(data), chunk_size):
#             await crud.save_to_db(data[i:i+chunk_size])

#         return {"message": "PDFs loaded successfully."}
    
#     except Exception as e:
#         logging.error(f"app.py: Error with loading PDFs: {e}")
#         return {"message": "Failed to load PDFs."}

@app.post("/upload_pdfs", dependencies=[Depends(require_roles(*CONTENT_ROLES))])
async def upload_multiple_pdfs(files: List[UploadFile] = File(...), collection_name: str = Form(...), user: str = Form(...)):
    """
    Handles the upload of multiple PDF files.
    """
    processed_files = []
    failed_files = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}. Only PDFs are allowed.")
        try:
            text = await read_pdf_text(file)
            file_embeddings = await generate_embedding(text)

            pdf = (text, file_embeddings)
            data = await crud.save_pdfs_from_discord(user, collection_name, pdf)

            chunk_size = 10
            for i in range(0, len(data), chunk_size):
                await crud.save_to_db(data[i:i+chunk_size])

            logging.info(f"PDFs saved successfully for user {user} to collection {collection_name}.")
            processed_files.append(file.filename)
        except Exception as e:
            logging.error(f"app.py: Error with uploading PDFs: {e}")
            failed_files.append({"filename": file.filename, "error": str(e)})
    
    if failed_files:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process {len(failed_files)} file(s): {failed_files}"
        )
    
    return {
        "message": "PDFs uploaded and saved successfully.",
        "collection": collection_name,
        "processed_count": len(processed_files),
        "processed_files": processed_files,
        "user": user
    }

@app.post('/grading_expert', dependencies=[Depends(require_roles(*CONTENT_ROLES))]) #, response_model=QueryResponse
async def grading_expert(file: UploadFile = File(...)):
    file_embeddings = None


    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}. Only PDFs are allowed.")

        text = await read_pdf_text(file)
        file_embeddings = await generate_embedding(text)
    
        if not file_embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate file embeddings.")
        
        
        # Use similarity search to pick the most relevant rubric items (prevents sending everything)
        similar = await crud.get_data_by_similarity("grading_expert", file_embeddings, top_k=5)
        rubric_texts = []
        try:
            docs = similar.get('documents', [])
            if isinstance(docs, list) and len(docs) > 0:
                rubric_texts = docs[0]
        except Exception as e:
            logging.info(f'Grading expert error: {e}')
            rubric_texts = []

        logging.info(f"Relevant rubric documents (top_k=5): {rubric_texts}")

        # Truncate submission if too long to avoid excessive token usage
        max_submission_length = 5000
        submission_text = text[:max_submission_length] + ("..." if len(text) > max_submission_length else "")

        # Keep submission in data dict for template rendering, not as the query
        combined_data = {
            "rubric": rubric_texts,
            "retrieved_context": rubric_texts,
            "submission": submission_text,
            "student_submission": submission_text
        }

        # Use a short query instead of the entire PDF text
        query = "Grade this student submission according to the rubric."
        answer = await fetchGptResponse(query, PROMPTS['grading_expert'], combined_data)        
        logging.info(f"Answer: {answer}")
        return {'answer': answer}

    except Exception as e:
        logging.error(f"Error grading submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query")
async def query_endpoint(
    body: QueryRequest,
    user: AuthenticatedUser = Depends(require_roles(*QUERY_ROLES)),
):
    # The role comes only from the validated token; there is no default.
    response = await semantic_router.process_query(body, role=user.role)
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
