from semantic_router import Route
from semantic_router.routers import SemanticRouter as AurelioSemanticRouter
from semantic_router.encoders import LocalEncoder
import os, sys, logging, inspect 

# Adding path to ensure utils and backend are detected
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utlis.config import OPENAI_API_KEY
from utlis.prompts import PROMPTS
from backend.modelsPydantic import QueryRequest
from databases.chroma.modelsChroma import generate_embedding
from services.queryLangchain import fetchGptResponse
from router.utterances import UTTERANCES
from fastapi import HTTPException

from router.RouteMap import ThreadSafeMap

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ROLE_ROUTES = {
    "student":["course_materials", "mental_support"],
    "professor":["course_materials","teaching_support"]
}

PROFILE_ROUTES_MAP={
    "course_materials":["material_info_rt"],
    "teaching_support":["progress_report","problem_solve_rt"],
    "mental_support":["mental_support_rt"]
}

class SemanticRouter:
    def __init__(self, crud):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # Set up variables
        self.crud = crud
        self.encoder = LocalEncoder()
        self.route_layer_cache = {} # Cache for route layers per role
        self.response_map_cache = {} # Cache for response maps per role
        self.routes = []
        self.route_layer = None

        # Define all possible route responses
        self.route_responses = {
            "material_info_rt": self.material_info_guidance,
            "progress_report": self.progress_report_guidance,
            "problem_solve_rt": self.problem_solve_guidance,
            "mental_support_rt": self.mental_support_guidance,
        }
 
    # Response functions
    async def progress_report_guidance(self, request=None):
        return "Tracking your submitted labs and reviewing feedback will help ensure steady progress."
 
    async def problem_solve_guidance(self, request=None):
        return "Let me help you work through this problem step by step."
 
    async def material_info_guidance(self, request=None):
        return "Here's the course material you requested."
 
    async def mental_support_guidance(self, request=None):
        return "I'm here to support your wellbeing. Let's talk about what's on your mind."

    def _init_route_layer(self):
        """Lazily initialize the underlying AurelioSemanticRouter."""
        if self.route_layer is None:
            if self.encoder is None:
                self.encoder = LocalEncoder()
            self.route_layer = AurelioSemanticRouter(encoder=self.encoder, routes=list(self.routes), auto_sync="local")
            if hasattr(self.route_layer, "sync"):
                try:
                    self.route_layer.sync("local")
                except Exception as e:
                    logging.error(f"Semantic Router Layer Startup Error: {e}")
                    raise RuntimeError(f"Failed to sync route layer: {e}") from e
 
    # has to change entirely
    def _setup_routes(self, routes: ThreadSafeMap):
        """Initialize routes and route layer"""
        router_routes = []
        for route, utterances in routes.items():
            route_obj = Route(name=route, utterances=utterances)
            router_routes.append(route_obj)
        # keep internal `self.routes` in sync with the route layer so
        # later calls to `add_route` append rather than overwrite.
        self.routes = router_routes
        # ensure the route layer exists before updating
        self._init_route_layer()
        self.route_layer.routes = list(self.routes)
        logging.info(f"Successfully setup routes: {self.route_layer.routes}")
        return {"status": "routes setup", "routes": [r.name for r in router_routes]}
    
    def add_route(self, name, utterances):
        # route_layer may be None; log current known route names
        current_routes = getattr(self.route_layer, 'routes', self.routes)
        logging.info(f"Before adding routes: {current_routes}\n")
        logging.info(f"name: {name}, utterances: {utterances}")
        route_obj = Route(name=name, utterances=utterances)
        self.routes.append(route_obj)
        # ensure router is initialized then update
        self._init_route_layer()
        self.route_layer.routes = self.routes
        if hasattr(self.route_layer, "sync"):
            try:
                self.route_layer.sync("local")
            except Exception as e:
                logging.error(f'Add Route Error: {e}')
        logging.info(f"Successfully added routes: {self.route_layer.routes}")


    async def fallback_response(self, request=None):
        return "I'm not sure I understood that. Could you rephrase or ask something more specific?"
        
    async def generate_expert_response(self, request, collection_name, prompt_name):
        """Generates response using LLM and relevant documents"""
        query_embedding = await generate_embedding(request.query)
        collection_name = collection_name
        relevant_docs = await self.crud.get_data_by_similarity(collection_name, query_embedding, top_k=5)

        # Extract the list of documents (Chroma returns list-of-lists)
        docs = relevant_docs.get('documents', [])
        content = docs[0] if isinstance(docs, list) and len(docs) > 0 else []
        logging.info(f"Relevant messages: {content}")

        # Pass only a compact dict to the LLM to avoid sending excessive data
        answer = await fetchGptResponse(request.query, PROMPTS[prompt_name], {"relevant_documents": content})

        logging.info(f"Answer: {answer}")
        return {'answer': answer}

    async def process_query(self, request: QueryRequest, role: str):
    """Process a query while enforcing role-based route access."""

    try:
        if not role:
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )

        allowed_routes = ROLE_ROUTES.get(role)

        if allowed_routes is None:
            raise HTTPException(
                status_code=403,
                detail="Invalid or unauthorized user role"
            )

        logging.info(
            "Processing query for authenticated role: %s", role
        )

        self._init_route_layer()

        route = self.route_layer(request.query)

        if hasattr(route, "name"):
            route_name = route.name
        elif isinstance(route, str):
            route_name = route
        elif isinstance(route, dict) and "name" in route:
            route_name = route["name"]
        else:
            route_name = str(route)

        # Enforce role-based route access
        if route_name not in allowed_routes:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this resource"
            )

        class RouteResult:
            def __init__(self, name):
                self.name = name

        return RouteResult(route_name)
    
    except HTTPException:
        raise

    except Exception as e:
        logging.exception("Error processing query: %s", request.query)
        return {"error": str(e)}    

# Factory function to create router instance
def create_router(crud):
    return SemanticRouter(crud)