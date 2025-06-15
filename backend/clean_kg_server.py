#!/usr/bin/env python3
"""
Clean FastAPI Knowledge Graph Ingestor
Based on Neo4j Labs LLM Graph Builder - Production Ready
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
    title="Knowledge Graph Builder API",
    description="Clean FastAPI service for creating knowledge graphs from documents using Neo4j",
    version="1.0.0"
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

# Initialize services
def get_neo4j_graph():
    """Get Neo4j graph connection"""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

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
        graph = get_neo4j_graph()
        result = graph.query("RETURN 'Connection successful' as message")
        return {
            "status": "success",
            "message": "Neo4j connection successful",
            "data": result
        }
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j connection failed: {str(e)}")

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and process a document into knowledge graph"""
    try:
        logger.info(f"Processing file: {file.filename}")
        
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
        
        # Store document metadata in Neo4j
        graph = get_neo4j_graph()
        
        # Create source document node
        create_source_query = """
        CREATE (d:Document {
            name: $filename,
            content: $content,
            chunks: $chunk_count,
            uploaded_at: datetime(),
            status: 'uploaded'
        })
        RETURN d
        """
        
        graph.query(create_source_query, {
            "filename": file.filename,
            "content": text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
            "chunk_count": len(docs)
        })
        
        return {
            "status": "success",
            "message": f"Document {file.filename} uploaded successfully",
            "chunks_created": len(docs),
            "file_size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/extract")
async def extract_knowledge_graph(
    file_name: str = Form(...),
    model: str = Form("gpt-4o"),
    allowed_nodes: Optional[str] = Form(None),
    allowed_relationships: Optional[str] = Form(None)
):
    """Extract knowledge graph from uploaded document"""
    try:
        logger.info(f"Extracting knowledge graph from: {file_name}")
        
        graph = get_neo4j_graph()
        llm = get_llm(model)
        
        # Get document content
        doc_query = "MATCH (d:Document {name: $filename}) RETURN d.content as content"
        result = graph.query(doc_query, {"filename": file_name})
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Document {file_name} not found")
        
        content = result[0]["content"]
        
        # Enhanced entity extraction prompt with specific instructions
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
        logger.info(f"LLM response: {response.content[:500]}...")
        
        try:
            # Clean the response content to extract just the JSON
            response_text = response.content.strip()
            
            # Find JSON boundaries
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx]
            kg_data = json.loads(json_text)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Raw response: {response.content}")
            
            # Enhanced fallback: create entities from content analysis
            # Simple keyword extraction as fallback
            import re
            
            # Extract proper nouns and common entity patterns
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
        
        # Create entities in Neo4j
        entity_count = 0
        created_entities = set()
        
        for entity in kg_data.get("entities", []):
            entity_name = entity["name"]
            entity_type = entity.get("type", "Entity")
            
            if entity_name in created_entities:
                continue
                
            create_entity_query = f"""
            MERGE (e:{entity_type} {{name: $name}})
            SET e += $properties
            WITH e
            MATCH (d:Document {{name: $filename}})
            MERGE (e)-[:MENTIONED_IN]->(d)
            RETURN e
            """
            
            try:
                graph.query(create_entity_query, {
                    "name": entity_name,
                    "properties": entity.get("properties", {}),
                    "filename": file_name
                })
                created_entities.add(entity_name)
                entity_count += 1
                logger.info(f"Created entity: {entity_name} ({entity_type})")
            except Exception as e:
                logger.error(f"Failed to create entity {entity_name}: {e}")
        
        # Create relationships
        rel_count = 0
        for rel in kg_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "RELATED_TO")
            
            if not source or not target:
                continue
                
            create_rel_query = f"""
            MATCH (a {{name: $source}}), (b {{name: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            RETURN r
            """
            
            try:
                result = graph.query(create_rel_query, {
                    "source": source,
                    "target": target,
                    "properties": rel.get("properties", {})
                })
                if result:
                    rel_count += 1
                    logger.info(f"Created relationship: {source} -[{rel_type}]-> {target}")
            except Exception as e:
                logger.error(f"Failed to create relationship {source} -> {target}: {e}")
        
        # Update document status
        update_doc_query = """
        MATCH (d:Document {name: $filename})
        SET d.status = 'processed',
            d.processed_at = datetime(),
            d.entity_count = $entity_count,
            d.relationship_count = $rel_count
        RETURN d
        """
        
        graph.query(update_doc_query, {
            "filename": file_name,
            "entity_count": entity_count,
            "rel_count": rel_count
        })
        
        return {
            "status": "success",
            "message": f"Knowledge graph extracted from {file_name}",
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
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/sources")
async def list_sources():
    """List all processed documents"""
    try:
        graph = get_neo4j_graph()
        query = """
        MATCH (d:Document)
        RETURN d.name as name, 
               d.status as status,
               d.chunks as chunks,
               d.entity_count as entities,
               d.relationship_count as relationships,
               d.uploaded_at as uploaded,
               d.processed_at as processed
        ORDER BY d.uploaded_at DESC
        """
        
        result = graph.query(query)
        
        return {
            "status": "success",
            "sources": result,
            "total": len(result)
        }
        
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {str(e)}")

@app.post("/query")
async def query_graph(
    query: str = Form(...),
    limit: int = Form(10)
):
    """Query the knowledge graph"""
    try:
        graph = get_neo4j_graph()
        
        # Simple query execution (be careful with user input in production)
        result = graph.query(query)
        
        return {
            "status": "success",
            "query": query,
            "results": result[:limit],
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.delete("/document/{file_name}")
async def delete_document(file_name: str):
    """Delete a document and its entities from the knowledge graph"""
    try:
        graph = get_neo4j_graph()
        
        # Delete document and related entities
        delete_query = """
        MATCH (d:Document {name: $filename})
        OPTIONAL MATCH (e)-[:MENTIONED_IN]->(d)
        DETACH DELETE e, d
        RETURN count(*) as deleted
        """
        
        result = graph.query(delete_query, {"filename": file_name})
        
        return {
            "status": "success",
            "message": f"Document {file_name} deleted successfully",
            "nodes_deleted": result[0]["deleted"] if result else 0
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

if __name__ == "__main__":
    # Check environment configuration
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY]):
        logger.error("Missing required environment variables")
        exit(1)
    
    logger.info("Starting Clean Knowledge Graph Builder API")
    uvicorn.run(app, host="0.0.0.0", port=7860)