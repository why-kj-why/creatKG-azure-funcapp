import os
from llama_index.core import Document
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.schema import TransformComponent,BaseNode

os.environ['NLTK_DATA'] = './nltk_data'
os.environ['TIKTOKEN_CACHE_DIR'] = './tiktoken_cache'


class MyGraphExtractor(TransformComponent):
    def __call__(
        self, llama_nodes: list[Document], **kwargs
    ) -> list[BaseNode]:
        for llama_node in llama_nodes:
            # be sure to not overwrite existing entities/relations

            existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

            existing_nodes.append(
                EntityNode(
                    name=llama_node.text, label=llama_node.metadata["type"], properties={key:value for (key,value) in llama_node.metadata.items() if key!="type"}
                )
            )
            # if llama_node.metadata["type"] == "SCHEMA":
            #     for table in llama_node.metadata["tables"]:
            #         existing_relations.append(
            #             Relation(
            #                 label="HAS_TABLE",
            #                 source_id=llama_node.text,
            #                 target_id=table,
            #                 properties={},
            #             )
            #         ) # comment out when you don't want schema -> table relationship
            if llama_node.metadata["type"]=="TABLE":
                for column in llama_node.metadata["columns"]:
                    existing_relations.append(
                            Relation(
                                label="HAS_COLUMN",
                                source_id=llama_node.text,
                                target_id=column,
                                properties={},
                            )
                        )
            # add back to the metadata
            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes


def extract_documents_from_semantic(semantic_dict):
    documents = []
    for schema_name,content in semantic_dict.items():
        table_list = content["tables"]
        schema_metadata = {
            "type": "SCHEMA",
            "definition": content["schema_definition"],
            "tables": []
        }
        for table in table_list:
            table_name,table_content = list(table.items())[0]
            table_metadata = {
                "type": "TABLE",
                "definition": table_content["table_definition"],
                "columns": []
            }
            for col in table_content["columns"]:
                column_metadata = {
                    "type":"COLUMN",
                    "definition": list(col.values())[0],
                    "data_type": list(col.values())[1],
                    "table": table_name
                }
                col_doc = Document(text=list(col.keys())[0], metadata=column_metadata)
                table_metadata["columns"].append(list(col.keys())[0])
                documents.append(col_doc)

            table_doc = Document(text=table_name, metadata=table_metadata)
            documents.append(table_doc)
            schema_metadata["tables"].append(table_name)
        schema_doc = Document(text=schema_name, metadata=schema_metadata)
        documents.append(schema_doc)
    return documents


def create_nodes(llama_nodes: list[Document]) -> list[BaseNode]:
    for llama_node in llama_nodes:
        # be sure to not overwrite existing entities/relations
        existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

        existing_nodes.append(
            EntityNode(
                name=llama_node.text, label=llama_node.metadata["type"],
                properties={key: value for (key, value) in llama_node.metadata.items() if key != "type"}
            )
        )
        # if llama_node.metadata["type"] == "SCHEMA":
        #     for table in llama_node.metadata["tables"]:
        #         existing_relations.append(
        #             Relation(
        #                 label="HAS_TABLE",
        #                 source_id=llama_node.text,
        #                 target_id=table,
        #                 properties={},
        #             )
        #         ) # comment out when you don't want schema -> table relationship
        if llama_node.metadata["type"] == "TABLE":
            for column in llama_node.metadata["columns"]:
                existing_relations.append(
                    Relation(
                        label="HAS_COLUMN",
                        source_id=llama_node.text,
                        target_id=column,
                        properties={},
                    )
                )
        # add back to the metadata
        llama_node.metadata[KG_NODES_KEY] = existing_nodes
        llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

    return llama_nodes
