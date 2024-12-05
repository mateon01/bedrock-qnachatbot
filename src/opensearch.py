from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
from typing import Dict, Any, List, Optional

class OpenSearchClient:
    def __init__(self, host: str, region: str, service: str = 'aoss'):
        """
        Initialize OpenSearch client with AWS authentication
        
        Args:
            host: OpenSearch domain endpoint (without https://)
            region: AWS region (e.g., 'us-east-1')
            service: AWS service name ('aoss' for serverless, 'es' for OpenSearch Service)
        """
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, region, service)

        self.client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )

    def create_index(self, index_name: str, body: Optional[Dict] = None) -> Dict:
        """
        Create a new index
        
        Args:
            index_name: Name of the index to create
            body: Optional index settings and mappings
        """
        return self.client.indices.create(index=index_name, body=body)

    def delete_index(self, index_name: str) -> Dict:
        """Delete an index"""
        return self.client.indices.delete(index=index_name)

    def index_document(self, index_name: str, document: Dict, id: Optional[str] = None) -> Dict:
        """
        Index a document
        
        Args:
            index_name: Name of the index
            document: Document to index
            id: Optional document ID
        """
        return self.client.index(
            index=index_name,
            body=document,
            id=id
        )

    def search(self, index_name: str, query: Dict) -> Dict:
        """
        Search documents
        
        Args:
            index_name: Name of the index to search
            query: OpenSearch query DSL
        """
        return self.client.search(
            index=index_name,
            body=query
        )

    def get_document(self, index_name: str, id: str) -> Dict:
        """
        Get a document by ID
        
        Args:
            index_name: Name of the index
            id: Document ID
        """
        return self.client.get(index=index_name, id=id)

    def update_document(self, index_name: str, id: str, body: Dict) -> Dict:
        """
        Update a document
        
        Args:
            index_name: Name of the index
            id: Document ID
            body: Update body
        """
        return self.client.update(
            index=index_name,
            id=id,
            body={'doc': body}
        )

    def delete_document(self, index_name: str, id: str) -> Dict:
        """
        Delete a document
        
        Args:
            index_name: Name of the index
            id: Document ID
        """
        return self.client.delete(index=index_name, id=id)

    def bulk(self, operations: List[Dict]) -> Dict:
        """
        Perform bulk operations
        
        Args:
            operations: List of bulk operations
        """
        return self.client.bulk(body=operations)
    

class OpenSearchVectorClient:
    def __init__(self, host: str, region: str, model_id: str,service: str = 'aoss'):  # Titan embedding dimension is 1024
        """
        Initialize OpenSearch client with vector search capabilities
        
        Args:
            host: OpenSearch endpoint
            region: AWS region
            embedding_model_id: Bedrock embedding model ID
            rerank_model_id: Bedrock rerank model ID
            service: AWS service ('aoss' or 'es')
            vector_dimension: Dimension of the embedding vectors
        """
        # Initialize OpenSearch client
        credentials = boto3.Session(region_name='us-wes').get_credentials()
        auth = AWSV4SignerAuth(credentials, region, service)
        self.model_id = model_id
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )


    def hybrid_search(self,
                     index_name: str,
                     query_text: str,
                     k: int = 5,
                     min_score: float = 0.5,
                     semantic_weight: float = 0.6) -> Dict:
        """
        Perform hybrid search combining vector similarity and text matching
        
        Args:
            index_name: Name of the index
            query_text: Text to search for
            k: Number of documents to return
            semantic_weight: Weight for vector similarity (0 to 1)
        """
        search_query = {
            "size": k,
            "min_score": min_score,
            "search_pipeline": {
                "phase_results_processors": [
                    {
                        "normalization-processor": {
                            "normalization": {
                                "technique": "min_max"
                            },
                            "combination": {
                                "technique": "arithmetic_mean",
                                "parameters": {
                                    "weights": [
                                        1-semantic_weight,
                                        semantic_weight
                                    ]
                                }
                            }
                        }
                    }
                ]
            },
            "_source": {
                "includes": ["contextual","body_chunk_default", "table_md", "image_base64","title"]
            },
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "body_chunk_default^3",
                                    "qr^2",
                                    "contextual^2",
                                    "title",
                                    "table_md"
                                ],
                                "type": "most_fields"
                            }
                        },
                        {
                            "neural": {
                                "contextual_embedding": {
                                    "query_text": query_text,
                                    "model_id": self.model_id,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        return self.client.search(
            index=index_name,
            body=search_query
        )