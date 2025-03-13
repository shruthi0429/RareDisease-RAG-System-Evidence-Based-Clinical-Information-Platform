# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Tuple
import gradio as gr
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import ServiceContext
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import chromadb
import os
import shutil
import time

def prepare_documents(merged_data: Dict) -> List[Document]:
    """Convert merged disease data into LlamaIndex documents"""
    documents = []

    for disease_name, data in merged_data.items():
        if data.get('disease_info'):
            disease_info = data['disease_info']
            disease_text = f"""
            Disease Name: {disease_name}
            Clinical Definition: {disease_info.get('definition', '')}

            Clinical Features:
            {json.dumps(disease_info.get('clinical_features', {}), indent=2)}

            Genetic Information:
            {json.dumps(disease_info.get('genetic_info', {}), indent=2)}

            Natural History:
            {json.dumps(disease_info.get('natural_history', {}), indent=2)}

            Epidemiology:
            {json.dumps(disease_info.get('epidemiology', {}), indent=2)}
            """

            disease_doc = Document(
                text=disease_text,
                metadata={
                    "disease": disease_name,
                    "type": "clinical_info",
                    "source": "Orphanet"
                }
            )
            documents.append(disease_doc)

        for paper in data.get('papers', []):
            paper_text = f"""
            Title: {paper['title']}

            Abstract:
            {paper['abstract']}

            Authors: {', '.join(paper['authors'])}
            Journal: {paper['journal']}
            Publication Date: {paper['publication_date'].get('year', '')}
            """

            paper_doc = Document(
                text=paper_text,
                metadata={
                    "disease": disease_name,
                    "type": "research_paper",
                    "paper_id": paper['paper_id'],
                    "source": "PubMed"
                }
            )
            documents.append(paper_doc)

    return documents

def setup_chroma_store(persist_dir: str = "./chroma_db", embedding_dimension: int = 768):
    """Setup ChromaDB with specific embedding dimension"""
    try:
        collection_name = f"rare_diseases_collection"

        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                time.sleep(2)
            except Exception as e:
                print(f"Directory cleanup warning: {e}")

        settings = chromadb.Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=persist_dir
        )

        chroma_client = chromadb.PersistentClient(path=persist_dir, settings=settings)

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "dimension": embedding_dimension}
        )

        vector_store = ChromaVectorStore(chroma_collection=collection)
        return vector_store, chroma_client

    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise

def initialize_rag_system(merged_data: Dict):
    """Initialize the RAG system with ChromaDB and ClinicalBERT embeddings"""
    try:

        client = OpenAI(api_key = "your Open AI API Key ")
        os.environ['OPENAI_API_KEY'] = client.api_key

        embed_model = HuggingFaceEmbedding(
            model_name="medicalai/ClinicalBERT",
            max_length=512,
            embed_batch_size=32,
        )

        # Set up ChromaDB
        vector_store, chroma_client = setup_chroma_store(embedding_dimension=768)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            node_parser=SentenceSplitter(
                chunk_size=384,
                chunk_overlap=50
            )
        )

        # Process documents
        documents = prepare_documents(merged_data)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=True
        )

        print(f"Verifying embedding storage:")
        print(f"Number of stored embeddings: {len(documents)}")

        return index, "System initialized successfully with persistent storage!"

    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return None, f"Error initializing system: {str(e)}"

def query_disease(index, query: str, disease_name: str = None) -> str:
    """Query the RAG system"""
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )

        if disease_name and disease_name != "All Diseases":
            formatted_query = f"""
            Regarding {disease_name}:
            {query}
            Please provide evidence-based clinical information.
            """

            response = query_engine.query(
                formatted_query,
                metadata_filters={"disease": disease_name}
            )
        else:
            formatted_query = f"""
            {query}
            Please provide evidence-based clinical information about relevant rare diseases.
            """
            response = query_engine.query(formatted_query)

        return str(response)

    except Exception as e:
        print(f"Query error details: {str(e)}")
        return f"Error during query: {str(e)}"

def create_gradio_interface():
    with open('RareDisease_data.json', 'r') as f:
        merged_data = json.load(f)

    disease_list = ["All Diseases"] + list(merged_data.keys())

    index, init_message = initialize_rag_system(merged_data)
    print(f"Initialization status: {init_message}")

    def handle_query(query, disease_name):
        if not index:
            return "System initialization failed. Please check the logs."
        return query_disease(index, query, disease_name)

    with gr.Blocks(title="Clinical Rare Disease Information System") as interface:
        gr.Markdown("# Clinical Rare Disease Information System")
        gr.Markdown("### Evidence-Based Clinical Decision Support")

        with gr.Row():
            disease_dropdown = gr.Dropdown(
                choices=disease_list,
                label="Select Disease",
                value="All Diseases"
            )

        with gr.Row():
            question_input = gr.Textbox(
                label="Clinical Query",
                placeholder="Ask about the disease...",
                lines=2
            )
            submit_button = gr.Button("Submit Query")

        answer_output = gr.Textbox(
            label="Clinical Insights",
            lines=10
        )

        gr.Examples(
            examples=[
                ["What are the main clinical manifestations?", "Fabry Disease"],
                ["Describe the genetic basis and inheritance pattern", "Gaucher Disease"],
                ["What are the current treatment approaches?", "All Diseases"],
            ],
            inputs=[question_input, disease_dropdown]
        )

        submit_button.click(
            handle_query,
            inputs=[question_input, disease_dropdown],
            outputs=[answer_output]
        )

    return interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)
