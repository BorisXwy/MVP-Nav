import json

class GoalGraphDecomposer:

    # def __init__(self, llm=None):
    #     self.llm = llm
    #     self.goalgraph_decomposed = {}

    def __init__(self, llm=None):
        self.llm = llm
        self.goalgraph_decomposed = {}

    def build_graph(self, objects, relations):
        graph = {
            'nodes': [{'id': obj} for obj in objects],
            'edges': [{'source': r['source'], 'target': r['target'], 'type': r['type']} for r in relations]
        }
        return graph

    def clean_edges(self, subgraphs):
        for subgraph in subgraphs.values():
            if 'nodes' not in subgraph:
                subgraph['nodes'] = []
            if 'edges' not in subgraph:
                subgraph['edges'] = []
            node_ids = {node['id'] for node in subgraph['nodes']}
            subgraph['edges'] = [edge for edge in subgraph['edges'] if edge['source'] in node_ids and edge['target'] in node_ids]

    def graph_to_text(self, graph):
        if 'nodes' not in graph:
            graph['nodes'] = []
        if 'edges' not in graph:
            graph['edges'] = []
        nodes = ', '.join([node['id'] for node in graph['nodes']])
        edges = ', '.join([f"{edge['source']} {edge['type']} {edge['target']}" for edge in graph['edges']])
        return f"Nodes: {nodes}. Edges: {edges}."

    # def goal_decomposition(self, goalgraph=None):
    #     prompt = (f"Given the following graph, decompose it into a set of subgraphs where each subgraph contains strongly related nodes. "
    #               f"Output the subgraphs in the same format as the input, with each subgraph having its own 'nodes' and 'edges' list. "
    #               f"The format should be: {{'subgraph_1': {{'nodes': [{{'id': 'node_id'}}], 'edges': [{{'source': 'source_node_id', 'target': 'target_node_id', 'type': 'relation_type'}}]}} , 'subgraph_2': {{...}}, ...}}. "
    #               f"Avoid including weakly related or unrelated nodes in the same subgraph. "
    #               f"Here is the graph to decompose: {self.graph_to_text(goalgraph)}")

    #     max_attempts = 10
    #     attempts = 0
    #     while attempts < max_attempts:
    #         response = self.llm(prompt)

    #         try:
    #             self.goalgraph_decomposed = json.loads(response)
    #             self.clean_edges(self.goalgraph_decomposed)
    #             break
    #         except json.JSONDecodeError:
    #             attempts += 1

    #     if attempts == max_attempts:
    #         self.goalgraph_decomposed = {'subgraph_1': goalgraph}

    #     return self.goalgraph_decomposed

    def goal_decomposition(self, goalgraph=None):
            print("--- goal_decomposition: start ---")
            prompt = (f"Given the following graph, decompose it into a set of subgraphs where each subgraph contains strongly related nodes. "
                    f"Output the subgraphs in the same format as the input, with each subgraph having its own 'nodes' and 'edges' list. "
                    f"The format should be: {{'subgraph_1': {{'nodes': [{{'id': 'node_id'}}], 'edges': [{{'source': 'source_node_id', 'target': 'target_node_id', 'type': 'relation_type'}}]}} , 'subgraph_2': {{...}}, ...}}. "
                    f"Avoid including weakly related or unrelated nodes in the same subgraph. "
                    f"Here is the graph to decompose: {self.graph_to_text(goalgraph)}")

            # max_attempts = 10
            max_attempts = 1
            attempts = 0
            print(f"goal_decomposition: calling LLM, max_attempts={max_attempts}.")

            while attempts < max_attempts:
                print(f"goal_decomposition: attempt {attempts + 1}, sending request...")
                response = self.llm(prompt)
                print(f"goal_decomposition: response received (attempt {attempts + 1}), preview: {response[:500]}...")

                try:
                    print(f"goal_decomposition: parsing JSON (attempt {attempts + 1})...")
                    self.goalgraph_decomposed = json.loads(response)
                    print("goal_decomposition: JSON OK, calling clean_edges...")
                    self.clean_edges(self.goalgraph_decomposed)
                    break
                except json.JSONDecodeError as e:
                    attempts += 1
                    print(f"goal_decomposition: JSON parse failed (attempt {attempts}): {e}")
                except Exception as e:
                    attempts += 1
                    print(f"goal_decomposition: error (attempt {attempts}): {e}")

            if attempts == max_attempts:
                print(f"goal_decomposition: max attempts ({max_attempts}) reached, using original goalgraph.")
                self.goalgraph_decomposed = {'subgraph_1': goalgraph}

            print("--- goal_decomposition: done ---")
            return self.goalgraph_decomposed
