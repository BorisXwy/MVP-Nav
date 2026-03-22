class GraphBuilder:
    def __init__(self, llm=None):
        self.llm = llm

    def get_objects(self, llm_response):
        objects = llm_response.split('[')[1].split(']')[0].split(',')
        for i in range(len(objects)):
            objects[i] = objects[i].strip().lower()
        return objects

    def get_relations(self, llm_response, objects):
        relations = []
        for line in llm_response.split('\n'):
            relation = {'source': '', 'target': '', 'type': ''}
            parts = line.strip().split(': ')
            if len(parts) == 2:
                relation_info = parts[1].strip()
                relation_parts = relation_info.split(' is ')
                if len(relation_parts) == 2:
                    source_target = parts[0].strip().split(' and ')
                    if len(source_target) == 2:
                        source = source_target[0].strip()
                        target = source_target[1].strip()
                        relation_type = relation_parts[1].strip()
                        if source in objects and target in objects:
                            relation = {'source': source, 'target': target, 'type': relation_type}
                            relations.append(relation)
        return relations

    def parse_text_description(self, description):
        object_prompt = f"""
        Please extract all objects mentioned in the following text description and the output format is "[<object 1>, <object 2>,...]":
        Text Description: {description}
        """
        object_response = self.llm(object_prompt)
        print("Object Response:", object_response) 
        main_objects = []
        sub_objects = []
        main_objects.append(self.get_objects(object_response)[0])
        for obj in self.get_objects(object_response)[1:]:
            sub_objects.append(obj)

        return main_objects, sub_objects

    def build_graph(self, objects, relations):
        graph = {
            'nodes': [{'id': obj} for obj in objects],
            'edges': [{'source': r['source'], 'target': r['target'], 'type': r['type']} for r in relations]
        }
        return graph

    def get_goal_from_text(self, text_goal):
        main_objects, sub_objects = self.parse_text_description(text_goal)
        # graph = self.build_graph(objects, relations)
        return main_objects, sub_objects
