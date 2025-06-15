#!/usr/bin/env python3
"""
Multi-Tenant Knowledge Graph Builder API
Based on Neo4j Labs LLM Graph Builder with Multi-KG Architecture
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import tempfile
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
from datetime import datetime

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Tenant Knowledge Graph Builder API",
    description="Multi-tenant FastAPI service for creating isolated knowledge graphs using Neo4j",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MultiKGManager:
    """Multi-tenant Knowledge Graph Manager"""
    
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Create performance indexes for multi-KG queries"""
        try:
            # Indexes for KG identification
            self.graph.query("CREATE INDEX kg_id_index IF NOT EXISTS FOR (n) ON (n.kg_id)")
            self.graph.query("CREATE INDEX kg_name_index IF NOT EXISTS FOR (n) ON (n.kg_name)")
            
            # Constraint for unique KG names
            self.graph.query("""
                CREATE CONSTRAINT kg_unique_name IF NOT EXISTS 
                FOR (kg:KnowledgeGraph) REQUIRE kg.name IS UNIQUE
            """)
            self.graph.query("""
                CREATE CONSTRAINT kg_unique_id IF NOT EXISTS 
                FOR (kg:KnowledgeGraph) REQUIRE kg.id IS UNIQUE
            """)
            logger.info("Multi-KG indexes and constraints created")
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")
    
    def create_kg(self, kg_id: str, kg_name: str, description: str = ""):
        """Create a new knowledge graph root"""
        query = """
        MERGE (kg:KnowledgeGraph {id: $kg_id})
        ON CREATE SET 
            kg.name = $kg_name,
            kg.description = $description,
            kg.created_at = datetime(),
            kg.version = "1.0",
            kg.status = "active"
        ON MATCH SET 
            kg.updated_at = datetime()
        RETURN kg
        """
        
        result = self.graph.query(query, {
            "kg_id": kg_id,
            "kg_name": kg_name, 
            "description": description
        })
        
        logger.info(f"Created/Updated KG: {kg_name} ({kg_id})")
        return result
    
    def get_kg_label(self, kg_id: str) -> str:
        """Convert KG ID to label format"""
        # Query the actual KG name from database
        query = "MATCH (kg:KnowledgeGraph {id: $kg_id}) RETURN kg.name as name"
        result = self.graph.query(query, {"kg_id": kg_id})
        
        if result:
            return result[0]["name"].replace(" ", "").replace("-", "")
        
        # Fallback mapping for common patterns
        label_map = {
            "finance_2024": "FinanceKG",
            "research_papers": "ResearchKG",
            "legal_documents": "LegalKG",
            "healthcare_docs": "HealthcareKG",
            "general": "GeneralKG"
        }
        return label_map.get(kg_id, f"KG_{kg_id.upper()}")
    
    def ingest_document(self, kg_id: str, filename: str, content: str, chunks: int):
        """Ingest document into specific knowledge graph"""
        # Ensure KG exists
        kg_name = self.get_kg_label(kg_id)
        self.create_kg(kg_id, kg_name, f"Knowledge graph for {kg_id}")
        
        # Create document with KG association
        query = f"""
        MATCH (kg:KnowledgeGraph {{id: $kg_id}})
        CREATE (doc:Document:{kg_name} {{
            name: $filename,
            content: $content,
            chunks: $chunks,
            kg_id: $kg_id,
            kg_name: kg.name,
            uploaded_at: datetime(),
            status: 'uploaded'
        }})
        CREATE (kg)-[:CONTAINS]->(doc)
        RETURN doc
        """
        
        result = self.graph.query(query, {
            "kg_id": kg_id,
            "filename": filename,
            "content": content[:1000] + "..." if len(content) > 1000 else content,
            "chunks": chunks
        })
        
        logger.info(f"Ingested document {filename} into KG {kg_id}")
        return result
    
    def create_entity(self, kg_id: str, entity_name: str, entity_type: str, properties: dict, filename: str):
        """Create entity in specific knowledge graph"""
        kg_name = self.get_kg_label(kg_id)
        
        query = f"""
        MATCH (kg:KnowledgeGraph {{id: $kg_id}})
        MERGE (e:Entity:{entity_type}:{kg_name} {{name: $name, kg_id: $kg_id}})
        ON CREATE SET 
            e.kg_name = kg.name,
            e.created_at = datetime(),
            e += $properties
        ON MATCH SET 
            e.updated_at = datetime(),
            e += $properties
        WITH e, kg
        MATCH (d:Document {{name: $filename, kg_id: $kg_id}})
        MERGE (e)-[:MENTIONED_IN]->(d)
        CREATE (kg)-[:CONTAINS]->(e)
        RETURN e
        """
        
        try:
            result = self.graph.query(query, {
                "kg_id": kg_id,
                "name": entity_name,
                "properties": properties,
                "filename": filename
            })
            return result
        except Exception as e:
            logger.error(f"Failed to create entity {entity_name} in KG {kg_id}: {e}")
            return None
    
    def create_relationship(self, kg_id: str, source: str, target: str, rel_type: str, properties: dict):
        """Create relationship between entities in specific knowledge graph"""
        query = f"""
        MATCH (a {{name: $source, kg_id: $kg_id}}), (b {{name: $target, kg_id: $kg_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $properties, r.kg_id = $kg_id
        RETURN r
        """
        
        try:
            result = self.graph.query(query, {
                "kg_id": kg_id,
                "source": source,
                "target": target,
                "properties": properties
            })
            return result
        except Exception as e:
            logger.error(f"Failed to create relationship {source} -> {target} in KG {kg_id}: {e}")
            return None
    
    def list_kgs(self):
        """List all knowledge graphs"""
        query = """
        MATCH (kg:KnowledgeGraph)
        OPTIONAL MATCH (kg)-[:CONTAINS]->(node)
        RETURN 
            kg.name as name,
            kg.id as id,
            kg.description as description,
            kg.created_at as created,
            kg.status as status,
            count(node) as node_count
        ORDER BY kg.created_at DESC
        """
        return self.graph.query(query)
    
    def query_kg(self, kg_id: str, query: str):
        """Query specific knowledge graph"""
        # Add KG filter to user query
        if kg_id and kg_id != "all":
            # Simple approach: add WHERE clause for kg_id
            if "WHERE" in query.upper():
                query = query.replace("WHERE", f"WHERE n.kg_id = '{kg_id}' AND")
            else:
                # Add WHERE clause before RETURN
                if "RETURN" in query.upper():
                    query = query.replace("RETURN", f"WHERE n.kg_id = '{kg_id}' RETURN")
        
        return self.graph.query(query)

# Initialize multi-KG manager
kg_manager = MultiKGManager()

def get_llm(model: str = "gpt-4o"):
    """Get OpenAI LLM instance"""
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/connect")
async def test_connection():
    """Test Neo4j database connection"""
    try:
        result = kg_manager.graph.query("RETURN 'Multi-tenant connection successful' as message")
        return {
            "status": "success",
            "message": "Multi-tenant Neo4j connection successful",
            "data": result
        }
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j connection failed: {str(e)}")

@app.get("/knowledge-graphs")
async def list_knowledge_graphs():
    """List all knowledge graphs in the system"""
    try:
        kgs = kg_manager.list_kgs()
        return {
            "status": "success",
            "knowledge_graphs": kgs,
            "total": len(kgs)
        }
    except Exception as e:
        logger.error(f"Failed to list KGs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list knowledge graphs: {str(e)}")

@app.post("/knowledge-graphs")
async def create_knowledge_graph(
    kg_id: str = Form(...),
    kg_name: str = Form(...),
    description: str = Form("")
):
    """Create a new knowledge graph"""
    try:
        result = kg_manager.create_kg(kg_id, kg_name, description)
        return {
            "status": "success",
            "message": f"Knowledge graph '{kg_name}' created successfully",
            "kg_id": kg_id,
            "kg_name": kg_name
        }
    except Exception as e:
        logger.error(f"Failed to create KG: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge graph: {str(e)}")

@app.post("/upload/{kg_id}")
async def upload_document(
    kg_id: str,
    file: UploadFile = File(...),
    model: str = Form("gpt-4o"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and process a document into specific knowledge graph"""
    try:
        logger.info(f"Processing file: {file.filename} for KG: {kg_id}")
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Create document chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        docs = text_splitter.create_documents([text_content])
        logger.info(f"Created {len(docs)} chunks from {file.filename}")
        
        # Ingest into specific KG
        result = kg_manager.ingest_document(kg_id, file.filename, text_content, len(docs))
        
        return {
            "status": "success",
            "message": f"Document {file.filename} uploaded to KG {kg_id}",
            "kg_id": kg_id,
            "chunks_created": len(docs),
            "file_size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/extract/{kg_id}")
async def extract_knowledge_graph(
    kg_id: str,
    file_name: str = Form(...),
    model: str = Form("gpt-4o"),
    allowed_nodes: Optional[str] = Form(None),
    allowed_relationships: Optional[str] = Form(None)
):
    """Extract knowledge graph from uploaded document in specific KG"""
    try:
        logger.info(f"Extracting knowledge graph from: {file_name} in KG: {kg_id}")
        
        llm = get_llm(model)
        
        # Get document content from specific KG
        doc_query = "MATCH (d:Document {name: $filename, kg_id: $kg_id}) RETURN d.content as content"
        result = kg_manager.graph.query(doc_query, {"filename": file_name, "kg_id": kg_id})
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Document {file_name} not found in KG {kg_id}")
        
        content = result[0]["content"]
        
        # Enhanced entity extraction prompt
        extraction_prompt = f"""
        You are an expert knowledge graph extractor. Analyze the following text and extract ALL entities and relationships.

        Text to analyze:
        {content}

        Extract entities and relationships following these guidelines:
        1. Identify PEOPLE (names, titles, roles)
        2. Identify ORGANIZATIONS (companies, institutions, hospitals)
        3. Identify LOCATIONS (cities, states, countries)
        4. Identify PRODUCTS (software, tools, platforms)
        5. Identify CONCEPTS (technologies, fields, specializations)
        6. Identify EVENTS (conferences, funding rounds, developments)

        For relationships, identify:
        - WORKS_FOR (person works for organization)
        - FOUNDED (person founded organization)
        - LOCATED_IN (organization located in location)
        - PARTNERED_WITH (organization partnered with organization)
        - DEVELOPED (organization developed product)
        - SPECIALIZES_IN (person/org specializes in concept)
        - PRESENTED_AT (organization presented at event)
        - LEADS (person leads team/department)
        - RAISED (organization raised funding)

        Return ONLY valid JSON in this exact format:
        {{
            "entities": [
                {{"name": "TechCorp", "type": "ORGANIZATION", "properties": {{"description": "Leading technology company", "founded": "2020"}}}},
                {{"name": "John Smith", "type": "PERSON", "properties": {{"role": "CEO", "education": "PhD Computer Science Stanford"}}}},
                {{"name": "San Francisco", "type": "LOCATION", "properties": {{"state": "California"}}}},
                {{"name": "MedAI Assistant", "type": "PRODUCT", "properties": {{"description": "AI-powered diagnostic tool"}}}}
            ],
            "relationships": [
                {{"source": "John Smith", "target": "TechCorp", "type": "WORKS_FOR", "properties": {{"role": "CEO"}}}},
                {{"source": "TechCorp", "target": "San Francisco", "type": "LOCATED_IN", "properties": {{}}}},
                {{"source": "TechCorp", "target": "MedAI Assistant", "type": "DEVELOPED", "properties": {{}}}}
            ]
        }}
        """
        
        # Get LLM response
        response = llm.invoke(extraction_prompt)
        logger.info(f"LLM response for KG {kg_id}: {response.content[:200]}...")
        
        try:
            # Parse JSON response
            response_text = response.content.strip()
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx]
            kg_data = json.loads(json_text)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing failed for KG {kg_id}: {e}")
            # Fallback pattern matching
            import re
            people_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            org_pattern = r'\b[A-Z][a-zA-Z]*(Corp|Inc|LLC|University|Hospital|Center|System)\b'
            
            people = list(set(re.findall(people_pattern, content)))
            orgs = list(set(re.findall(org_pattern, content)))
            
            kg_data = {
                "entities": [
                    {"name": person, "type": "PERSON", "properties": {}} for person in people[:10]
                ] + [
                    {"name": org, "type": "ORGANIZATION", "properties": {}} for org in orgs[:10]
                ],
                "relationships": [
                    {"source": people[0], "target": orgs[0], "type": "WORKS_FOR", "properties": {}}
                ] if people and orgs else []
            }
        
        # Create entities in specific KG
        entity_count = 0
        created_entities = set()
        
        for entity in kg_data.get("entities", []):
            entity_name = entity["name"]
            entity_type = entity.get("type", "Entity")
            
            if entity_name in created_entities:
                continue
            
            result = kg_manager.create_entity(
                kg_id, entity_name, entity_type, 
                entity.get("properties", {}), file_name
            )
            
            if result:
                created_entities.add(entity_name)
                entity_count += 1
                logger.info(f"Created entity in KG {kg_id}: {entity_name} ({entity_type})")
        
        # Create relationships in specific KG
        rel_count = 0
        for rel in kg_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target") 
            rel_type = rel.get("type", "RELATED_TO")
            
            if not source or not target:
                continue
            
            result = kg_manager.create_relationship(
                kg_id, source, target, rel_type, rel.get("properties", {})
            )
            
            if result:
                rel_count += 1
                logger.info(f"Created relationship in KG {kg_id}: {source} -[{rel_type}]-> {target}")
        
        # Update document status in specific KG
        update_doc_query = """
        MATCH (d:Document {name: $filename, kg_id: $kg_id})
        SET d.status = 'processed',
            d.processed_at = datetime(),
            d.entity_count = $entity_count,
            d.relationship_count = $rel_count
        RETURN d
        """
        
        kg_manager.graph.query(update_doc_query, {
            "filename": file_name,
            "kg_id": kg_id,
            "entity_count": entity_count,
            "rel_count": rel_count
        })
        
        return {
            "status": "success",
            "message": f"Knowledge graph extracted from {file_name} in KG {kg_id}",
            "kg_id": kg_id,
            "entities_created": entity_count,
            "relationships_created": rel_count,
            "model_used": model,
            "extraction_details": {
                "total_entities_found": len(kg_data.get("entities", [])),
                "total_relationships_found": len(kg_data.get("relationships", [])),
                "entities_successfully_created": entity_count,
                "relationships_successfully_created": rel_count
            }
        }
        
    except Exception as e:
        logger.error(f"Extraction failed for KG {kg_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/sources/{kg_id}")
async def list_sources(kg_id: str):
    """List all processed documents in specific knowledge graph"""
    try:
        if kg_id == "all":
            query = """
            MATCH (d:Document)
            RETURN d.name as name, 
                   d.kg_id as kg_id,
                   d.kg_name as kg_name,
                   d.status as status,
                   d.chunks as chunks,
                   d.entity_count as entities,
                   d.relationship_count as relationships,
                   d.uploaded_at as uploaded,
                   d.processed_at as processed
            ORDER BY d.uploaded_at DESC
            """
            result = kg_manager.graph.query(query)
        else:
            query = """
            MATCH (d:Document {kg_id: $kg_id})
            RETURN d.name as name, 
                   d.kg_id as kg_id,
                   d.kg_name as kg_name,
                   d.status as status,
                   d.chunks as chunks,
                   d.entity_count as entities,
                   d.relationship_count as relationships,
                   d.uploaded_at as uploaded,
                   d.processed_at as processed
            ORDER BY d.uploaded_at DESC
            """
            result = kg_manager.graph.query(query, {"kg_id": kg_id})
        
        return {
            "status": "success",
            "kg_id": kg_id,
            "sources": result,
            "total": len(result)
        }
        
    except Exception as e:
        logger.error(f"Failed to list sources for KG {kg_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {str(e)}")

@app.post("/query/{kg_id}")
async def query_knowledge_graph(
    kg_id: str,
    query: str = Form(...),
    limit: int = Form(10)
):
    """Query specific knowledge graph"""
    try:
        if kg_id == "all":
            # Query across all KGs
            result = kg_manager.graph.query(query)
        else:
            # Query specific KG
            result = kg_manager.query_kg(kg_id, query)
        
        return {
            "status": "success",
            "kg_id": kg_id,
            "query": query,
            "results": result[:limit],
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Query failed for KG {kg_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    # Check environment configuration
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY]):
        logger.error("Missing required environment variables")
        exit(1)
    
    logger.info("Starting Multi-Tenant Knowledge Graph Builder API")
    uvicorn.run(app, host="0.0.0.0", port=7860)