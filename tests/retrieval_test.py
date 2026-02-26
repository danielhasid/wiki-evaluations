import time

from adapters.retrieval_adapter import (
    MilvusConnector,
    retrieve_documents_and_scores_with_llm_keywords,
    retrieve_documents_and_scores_with_bm25_keywords,
    retrieve_documents_and_scores_with_vectordb_semantic,
    filter_docs_by_scores,
    retrieve_relevant_docs,
)


if __name__ == '__main__':
    query = "What product did Binh Nguyen advertise on 10/02/2019 15:52:40?"
    agent_name = "A06_OSINT"
    vector_db = MilvusConnector(collection_name=agent_name).get_milvus_client()

    # docs = retrieve_documents_and_scores_with_vectordb_semantic(query=query, vector_db=vector_db, agent_name=agent_name,
    #                                                             k=1)

    docs = retrieve_relevant_docs(query=query, vector_db=vector_db, agent_name=agent_name, k=5)

    # question_he = "אילו טילים יש לאיראן?"
    # agent_name = "Hatam_INSS"
    # # vector_db = MilvusConnector(collection_name=agent_name).get_milvus_client()
    # retrieve_documents_and_scores_with_llm_keywords(query=question_he, agent_name=agent_name, k=20)
    #
    exit()
