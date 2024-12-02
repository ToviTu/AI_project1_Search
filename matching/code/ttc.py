def ttc(patientPreference, kidneyPreference):
    patientMatch = {}  # patient id to kidney
    """IMPLEMENT METHOD HERE"""

    # Why kidneyPreference??? It seems not necessary
    patientPreference_ = patientPreference.copy()
    while patientPreference_:

        preference_graph = (
            {}
        )  # from a node (group of patient and kidney) to another node

        # build the preference graph
        to_delete = []
        for patient in patientPreference_:
            # if there is no compatible kidney
            if not patientPreference_[patient]:
                to_delete.append(patient)
                continue

            # find the next unmatched
            idx = 0
            while (
                idx < len(patientPreference_[patient]) - 1
                and patientPreference_[patient][idx] in patientMatch
            ):
                idx += 1
            preference_graph[patient] = patientPreference_[patient][idx]

        # cycle detection
        nodes_to_check = set(preference_graph.keys())
        get_next_node = lambda x: (
            next_node
            if (next_node := preference_graph.get(x, None)) not in patientMatch
            else None
        )
        at_least_one_cycle = False
        while nodes_to_check:
            node = nodes_to_check.pop()
            path = [node]
            has_cycle = False
            next_node = get_next_node(node)
            while next_node:
                if next_node in path:
                    has_cycle = True
                    at_least_one_cycle = True
                    path = path[path.index(next_node) :]
                    break
                path.append(next_node)
                next_node = get_next_node(next_node)

            # remove cycle (match)
            if has_cycle:
                node = path[0]
                next_node = preference_graph[node]
                patientMatch[node] = next_node  # match
                patientPreference_.pop(node)  # remove

                if node in nodes_to_check:
                    nodes_to_check.remove(node)

                while next_node != node:
                    patientMatch[next_node] = preference_graph[next_node]  # match
                    patientPreference_.pop(next_node)  # remove
                    nodes_to_check.remove(next_node)

                    next_node = preference_graph.pop(next_node)

        # no more cycle
        if not at_least_one_cycle:
            break

    return patientMatch


