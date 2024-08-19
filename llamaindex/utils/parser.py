from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Node, RelationshipType, NodeRelationship


class CustomCSVParser(SimpleNodeParser):
    def get_nodes_from_documents(self, documents):
        nodes = []
        relationships = []

        for doc in documents:
            if "index_matching" in doc.metadata["file_name"]:
                # Handle index matching tables
                for _, row in doc.to_pandas().iterrows():
                    relationships.append(
                        NodeRelationship(
                            source_node_id=f"node_{row['idx_table_1']}",
                            target_node_id=f"node_{row['idx_table_2']}",
                            relationship_type=RelationshipType.RELATED,
                            metadata={"confidence": row["conf_values"]},
                        )
                    )
            else:
                # Handle direct entry tables
                for _, row in doc.to_pandas().iterrows():
                    node = Node(
                        text=f"{row['Name']} works at {row['Company']}",
                        id_=f"node_{_}",
                        metadata={"name": row["Name"], "company": row["Company"]},
                    )
                    nodes.append(node)

        return nodes, relationships


# Use the custom parser when creating the index
custom_parser = CustomCSVParser()
# index = KnowledgeGraphIndex.from_documents(
#     documents, storage_context=storage_context, node_parser=custom_parser
# )
