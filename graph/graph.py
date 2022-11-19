import math
import os
import sys
from graph.visualizer import GraphVisualizer
import hedwig

class Graph:
    def __init__(self, numIn : int, numOut : int, config) -> None:
        assert numIn > 0, 'Graph needs more then 0 input nodes'
        assert numOut > 0, 'Graph needs more then 0 output nodes'

        self.neighbourhood_size = config["neighbourhood_size"]
        self.next_node_ID = numIn + numOut + 1
        self.adj = {} # Connection list id -> id
        self.rev_adj = {}
        self.lookup = {} # Partial ordering lookup id -> order
        self.ordering = { 0: {}, 1: {} }#, 2: {}} # Ordering table
        self.new_nodes = {} # Lookup to check for dupplicate additions (Need resetting!)
        self.probability = {} #Lookup (from, to) connection to (probability, True), where True is to keep and False is to remove and probability is NCA output

        self.acts = {}
        self.bias = {}
        self.weights = {}

        self.weight_index = pow(self.neighbourhood_size,2) - 1
        self.acts_index = self.weight_index * 2
        self.bias_index = self.acts_index + 10

        self.start_weight = config["default_weight"]
        self.start_act = config["default_act"]
        self.start_bias = config["default_bias"]
        self.output_act = config["output_activation"]
        self.add_conn_threshold = config["add_conn_threshold"]
        self.add_node_threshold = config["add_node_threshold"]
        self.remove_conn_threshold = config["remove_conn_threshold"]
        self.remove_dead_conn_threshold = config["remove_dead_conn_threshold"]

        # Fill id's of start structure
        self.input = list(range(1, numIn + 1)) # Start from 1
        self.hidden = []
        self.output = list(range(numIn + 1, numIn + numOut + 1))

        #Center output around input
        output_suborder_start_index = 0
        if config["center_output_around_input"]: #False in case of legacy AI's which were evolved before this feature
            o_len = len(self.output)
            i_len = len(self.input)
            diff = abs(i_len - o_len)
            offset = int(diff/2)
            output_suborder_start_index = offset if i_len > o_len else -offset
        
        # Make the fully connected stucture
        counter = 0
        for node in self.input:
            self.weights[node, numIn + numOut + 1] = self.start_weight
            self.bias[node] = self.start_bias
            self.acts[node] = self.start_act
            self.rev_adj[node] = []
            self.adj[node] = []
            for other in self.output:
                self.adj[node].append(other)
                self.weights[node, other] = self.start_weight
            
            self.lookup[node] = (0, counter)
            self.ordering[0][counter] = node
            counter += 1

        counter = output_suborder_start_index
        for node in self.output:
            self.bias[node] = self.start_bias
            self.acts[node] = self.output_act
            self.rev_adj[node] = []
            for other in self.input:
                self.rev_adj[node].append(other)
                
            self.lookup[node] = (1, counter)
            self.ordering[1][counter] = node
            counter += 1
            
        self.max_x = 1
        self.max_y = max([len(self.input), len(self.output)])
    
    def prune_islands(self) -> None:
        marked = {} #Dict from id to mark
        def rec(node_id: int) -> bool:
            if node_id in self.output:
                marked[node_id] = True
                return True
            
            reachedOutput = False
            for next_id in self.adj[node_id]:
                if next_id not in marked:
                    reachedOutput = rec(next_id) or reachedOutput
                else:
                    reachedOutput = reachedOutput or marked[next_id]
            marked[node_id] = reachedOutput
            return reachedOutput
        
        for id in self.input:
            rec(id)
            marked[id] = True
            
        self.hidden = [id for id in self.hidden if marked.get(id, False)]
        self.adj = {id:[next_id for next_id in ids if marked.get(next_id, False)] for (id, ids) in self.adj.items() if marked.get(id, False)}

    def get_ids(self) -> 'list[int]':
        # Returns all id's of mutatable nodes in the graph
        return self.hidden.copy()

    def get_inputs(self) -> 'list[int]':
        return self.input.copy()
        
    def get_neighbourhood(self, id : int) -> 'list[float]':
        pos = self.lookup[id]
        nodes = [] # Neighbourhood nodes
        conn = [] # Connections to other neighbourhood nodes
        weights = [] #Neighbourhood weights

        # Lookup all neighbours
        neighbourhood_range = (int) ((self.neighbourhood_size - 1)/2)
        for i in range(-neighbourhood_range, neighbourhood_range + 1):
            for j in range(-neighbourhood_range, neighbourhood_range + 1):
                if i == 0 and j == 0: # Remove self
                    continue

                # Calculate global position
                dx = pos[0] + i
                dy = pos[1] + j

                # Lookup if node does exist
                if dx in self.ordering and dy in self.ordering[dx]:
                    otherID = self.ordering[dx][dy]
                    nodes.append(1.)

                    # Check for connections between nodes
                    if otherID in self.adj and id in self.adj[otherID]:
                        conn.append(1.) # A connection from other -> this
                        weights.append(self.weights[otherID, id])
                    elif id in self.adj and otherID in self.adj[id]:
                        conn.append(1.) # A connection from this -> other
                        weights.append(self.weights[id, otherID])
                    else:
                        conn.append(0.) # No connection found
                        weights.append(0.)
                else:
                    # No node found on position
                    nodes.append(0.)
                    conn.append(0.)
                    weights.append(0.)

        #Onehot encode activation function
        act_onehot = [0] * 10
        act_onehot[self.acts[id]] = 1
        return nodes + conn + weights + act_onehot + [self.bias[id]]

    def get_dead_connection_neighbourhood(self, _from, _to, iterations):
        max_incoming = self.neighbourhood_size * iterations * (1 - self.add_conn_threshold) * self.remove_conn_threshold * 2
        
        to_neuron_type = [int(_to in self.input), int(_to in self.hidden), int(_to in self.output)]
        from_neuron_type = [int(_from in self.input), int(_from in self.hidden), int(_from in self.output)]
        to_num_out = [0 if _to in self.output else len(self.adj[_to])/max_incoming]
        from_num_out = [len(self.adj[_from])/max_incoming]
        to_num_in = [len(self.rev_adj[_to])/max_incoming]
        from_num_in = [len(self.rev_adj[_from])/max_incoming]
        to_order, to_suborder = self.lookup[_to]
        from_order, from_suborder = self.lookup[_from]
        dx = [abs(from_order - to_order)/self.max_x]
        dy = [abs(from_suborder - to_suborder)/self.max_y]
        return to_neuron_type + from_neuron_type + to_num_out + from_num_out + to_num_in + from_num_in + dx + dy
        
    def update_dead_connection(self, _from, _to, dead_conn_output):
        if dead_conn_output[0] < self.remove_dead_conn_threshold:
            self.adj[_from].remove(_to)
            self.rev_adj[_to].remove(_from)
        else:
            self.weights[_from, _to] = dead_conn_output[1]

    def update_graph(self, id : int, arr : 'list[float]') -> None:
        if id in self.input:
            size = int((pow(self.neighbourhood_size, 2) - 1)/2 + (self.neighbourhood_size - 1)/2)
            arr[:size] = [0] * size

        pos = self.lookup[id]
        conn = arr[:self.weight_index]
        weights = arr[self.weight_index:self.acts_index]
        act_onehot = list(arr[self.acts_index:self.bias_index])

        self.acts[id] = act_onehot.index(max(act_onehot))
        self.bias[id] = arr[self.bias_index]

        
        counter = 0
        neighbourhood_range = (int) ((self.neighbourhood_size - 1)/2)
        for i in range(-neighbourhood_range, neighbourhood_range + 1):
            for j in range(-neighbourhood_range, neighbourhood_range + 1):
                if i == 0 and j == 0: # Remove self
                    continue

                # Calculate global position
                dx = pos[0] + i
                dy = pos[1] + j
                val = conn[counter]
                weight = weights[counter]
                counter += 1

                other_ID = -1

                before = int((pow(self.neighbourhood_size,2) - 1)/2) #Id's of nodes before target node
                if dx in self.ordering and dy in self.ordering[dx]:
                    # Using existing node
                    other_ID = self.ordering[dx][dy]

                    if counter <= before:
                        self.weights[other_ID, id] = weight
                    else:
                        self.weights[id, other_ID] = weight

                    if val < self.remove_conn_threshold: # Remove connection
                        probability = self.remove_conn_threshold - val
                        if counter <= before: # Other is before this
                            current_probability = self.probability.get((other_ID, id))
                            if current_probability is not None and current_probability[1] and probability < current_probability[0]:
                                continue
                            else:
                                self.probability[(other_ID, id)] = (probability, False)
                            if id in self.adj[other_ID]:
                                self.adj[other_ID].remove(id)
                            if other_ID in self.rev_adj[id]:
                                self.rev_adj[id].remove(other_ID)
                        else: # Other is after this
                            current_probability = self.probability.get((id, other_ID))
                            if current_probability is not None and current_probability[1] and probability < current_probability[0]:
                                continue
                            else:
                                self.probability[(id, other_ID)] = (probability, False)
                            if other_ID in self.adj[id]:
                                self.adj[id].remove(other_ID)
                            if id in self.rev_adj[other_ID]:
                                self.rev_adj[other_ID].remove(id)
                    elif val > self.add_conn_threshold: # Create connection to node 
                        probability = val - self.add_conn_threshold
                        if counter <= before: # Other is before this
                            current_probability = self.probability.get((other_ID, id))
                            if current_probability is not None and not current_probability[1] and probability < current_probability[0]:
                                continue
                            else:
                                self.probability[(other_ID, id)] = (probability, True)
                            if id not in self.adj[other_ID]:
                                self.adj[other_ID].append(id)
                            if other_ID not in self.rev_adj[id]:
                                self.rev_adj[id].append(other_ID)
                        else: # Other is after this
                            current_probability = self.probability.get((id, other_ID))
                            if current_probability is not None and not current_probability[1] and probability < current_probability[0]:
                                continue
                            else:
                                self.probability[(id, other_ID)] = (probability, True)
                            if other_ID not in self.adj[id]:
                                self.adj[id].append(other_ID)
                            if id not in self.rev_adj[other_ID]:
                                self.rev_adj[other_ID].append(id)
                else:
                    # Build a new node
                    if val > self.add_node_threshold:
                        node_id = self.next_node_ID

                        # new node is a dupplicate
                        if (dx, dy) in self.new_nodes:
                            node_id = self.new_nodes[(dx, dy)]
                        else:
                            self.adj[node_id] = []
                            self.rev_adj[node_id] = []
                            self.weights[node_id] = {}
                            self.new_nodes[(dx, dy)] = node_id
                            self.lookup[node_id] = (None, dy)
                            self.bias[node_id] = self.start_bias
                            self.acts[node_id] = self.start_act
                            self.hidden.append(node_id)
                            self.next_node_ID += 1
                        

                        # Add connection from this to new node
                        if counter <= before:
                            self.adj[node_id].append(id)
                            self.rev_adj[id].append(node_id)
                            self.weights[node_id, id] = weight
                        else:
                            self.adj[id].append(node_id)
                            self.rev_adj[node_id].append(id)
                            self.weights[id, node_id] = weight
                        
    def points_to_self(self):
        for id, lst in self.adj.items():
            if id in lst:
                hedwig.warning(f"{id} points to itself!!")


    def nagini_correction(self):
        self.__rebuild_partial_ordering()
        self.new_nodes.clear()
        self.probability.clear()

    def apply_cold_water(self):
        output_order = len(self.ordering) - 1
        self.ordering[len(self.ordering)].pop()

        while(len(self.ordering[output_order]) == 0):
            self.ordering[output_order].pop()
            output_order -= 1

        output_order += 1
        self.ordering[output_order] = {}
        counter = 0
        for id in self.output:
            self.ordering[output_order][counter] = id
            self.lookup[id] = (output_order, counter)   
        
    def fully_connect_output(self):
        for id in self.ordering[len(self.ordering) - 2].values():
            for otherID in self.output:
                if otherID not in self.adj[id]:
                    self.adj[id].append(otherID)
                    self.weights[id, otherID] = self.start_weight

    def find_dead_connections(self):
        dead_connections = []
        max_length = (self.neighbourhood_size - 1)/2
        for id, adj_lst in self.adj.items():
            id_order, id_pos = self.lookup[id]
            for adj in adj_lst:
                adj_order, adj_pos = self.lookup[adj]
                if abs(id_order - adj_order) > max_length or abs(id_pos - adj_pos) > max_length:
                    dead_connections.append((id, adj))
        return dead_connections

            
    def get_all_connections(self):
        return [(id, adj) for id, adj_lst in self.adj.items() for adj in adj_lst]

    def lux_fixes_everything(self, order_dic):
        main_nodes = list(order_dic.keys())
        flatten_adj = []
        for ids in self.adj.values():
            flatten_adj.extend(ids)
        
        no_order_no_in = [id for id in self.hidden if id not in flatten_adj and id not in order_dic]
        
        def forward(node, current_order):
            active_lst = set([node])
            next_lst = set()

            # Update the ordering for each node in the graph
            while len(active_lst) != 0:
                for id in active_lst:
                    org_order = order_dic.get(id, None)
                    if org_order is not None and org_order > current_order:
                        continue
                    order_dic[id] = current_order
                    for next in self.adj[id]:
                        if next not in self.output and next not in main_nodes:                   
                            next_lst.add(next)
                
                active_lst, next_lst = next_lst, set()
                current_order += 1

        
        def backward(node):
            if node in order_dic:
                return order_dic[node] -1
            if node in self.output:
                return sys.maxsize
            
            order = sys.maxsize
            for next_node in self.adj[node]:
                new_order = backward(next_node)
                if new_order < order:
                    order = new_order
            
            org_order = order_dic.get(node, sys.maxsize)
            if (org_order == sys.maxsize and order < sys.maxsize) or (org_order < sys.maxsize and order < org_order):
                forward(node, order)
                return order - 1
            
            return org_order
        
        org_order_dic_len = None
        while True:
            org_order_dic_len = len(order_dic)
            for node in no_order_no_in:
                backward(node)
            if len(order_dic) == org_order_dic_len:
                break
            
        for id in [node for node in self.hidden if node not in order_dic]:
            self.hidden.remove(id)
            self.adj.pop(id)
            
        for id in [node for node in order_dic if node not in self.input and order_dic[node] <= 0]:
            self.hidden.remove(id)
            self.adj.pop(id)
            order_dic.pop(id)
            
        return order_dic
            
        
  
    def __rebuild_partial_ordering(self) -> None:
        current_order = 0

        new_lookup = {}
        new_ordering = {}

        active_lst = set()
        active_lst |= set(self.input)
        next_lst = set()
        order_dic = {}

        unordered_nodes = self.hidden.copy()

        # Update the ordering for each node in the graph
        while len(active_lst) != 0:
            if current_order % 1000000 == 0: #To check if a node ever points to itself
                self.points_to_self()
            for id in active_lst:
                order_dic[id] = current_order
                if id in unordered_nodes:
                    unordered_nodes.remove(id)
                for next in self.adj[id]:
                    if next not in self.output:                   
                       next_lst.add(next)
            
            active_lst, next_lst = next_lst, set()
            current_order += 1

        #order_dic = self.partial_order_unconnected_to_input(unordered_nodes, order_dic)
        order_dic = self.lux_fixes_everything(order_dic)

        static_lookup = {} # lookup table for nodes that have not moved ordering position
        for id, order in order_dic.items(): # Calculation of suborders for each node
            old_order, old_pos = self.lookup[id]
            if order not in new_ordering:
                new_ordering[order] = {}

            # The node has not been moved
            # It should then have the same suborder
            if old_order == order:
                static_lookup[id] = True
                # If the space is free, add old node back
                if old_pos not in new_ordering[order]:
                    new_ordering[order][old_pos] = id
                    new_lookup[id] = (order, old_pos)
                # Space is not free, move the placed node down
                else:
                    node_id = new_ordering[order][old_pos]
                    node_pos = old_pos

                    # Override the node with the static node
                    new_ordering[order][old_pos] = id
                    new_lookup[id] = (order, old_pos)

                    # Try to move the new node down
                    while(True):
                        node_pos += 1
                        # Found empty space, place it.
                        if node_pos not in new_ordering[order]:
                            new_ordering[order][node_pos] = node_id
                            new_lookup[node_id] = (order, node_pos)
                            break
                        else:
                            # Swap if non-static node is found
                            other_id = new_ordering[order][node_pos]
                            if not static_lookup[other_id]:
                                new_ordering[order][node_pos] = node_id
                                new_lookup[node_id] = (order, node_pos)
                                node_id = other_id
            # Node has been moved since last run, attempting to place in same suborder, otherwise move it down.
            else:
                static_lookup[id] = False
                current_id = id
                current_pos = old_pos

                # Continue to run until free position is found
                while True:
                    # Place node if sub position is not in use already
                    if current_pos not in new_ordering[order]:
                        new_ordering[order][current_pos] = current_id
                        new_lookup[current_id] = (order, current_pos)
                        break # Since node is placed, stop the loop
                    # If position is already filled
                    else:
                        other_id = new_ordering[order][current_pos] # Lookup other node
                        if not static_lookup[other_id]: # Only swap node if is non-static
                            new_ordering[order][current_pos] = current_id
                            new_lookup[current_id] = (order, current_pos)
                            current_id = other_id
                    current_pos += 1 # Increment until free position is found

        # Construction of the output order layer
        output_order = len(new_ordering)
        new_ordering[output_order] = {}
        new_ordering[output_order] = self.ordering[len(self.ordering) - 1]
        for id in self.output:
            new_lookup[id] = (output_order, self.lookup[id][1])

        self.ordering = new_ordering
        self.lookup = new_lookup 

        self.max_x = max(list(self.ordering.keys()))
        max_y = 0
        for i in self.ordering:
            _min = min(self.ordering[i])
            _max = max(self.ordering[i])
            new_max_y = abs(_max - _min)
            if new_max_y > max_y:
                max_y = new_max_y
        self.max_y = max_y

    def get_graph(self) -> 'tuple[dict[int, list[int]], list[int], list[int], list[int], dict[int, int], dict[int, int], dict[dict[int, float]]]':
        # Returns the representation of the graph
        # 0 - Adjacency list
        # 1 - Input ids
        # 2 - Hidden ids
        # 4 - Output ids
        # 5 - Activation functions
        # 6 - Biases
        # 7 - Weights
        return (self.adj.copy(), self.input.copy(), self.hidden.copy(), self.output.copy(), self.acts.copy(), self.bias.copy(), self.weights.copy())
    
    def get_num_unconnected_nodes(self) -> int: #TODO new name
        unconnect_penalty = 0
        for i in self.input:
            if len(self.adj[i]) == 0:
                unconnect_penalty += 1
        for i in self.output:
            if len(self.rev_adj[i]) == 0:
                unconnect_penalty += 1
        return unconnect_penalty
    
    def any_hidden_nodes(self) -> bool:
        return len(self.hidden) != 0

    def draw(self, show, save) -> None:
        if not show and not save:
            return
        
        G = GraphVisualizer()

        for node_id in self.input + self.output + self.hidden:
            G.add_node(node_id, self.lookup[node_id], self.acts[node_id] + 1, self.bias[node_id])
            if node_id in self.output:
                continue
            for adj_id in self.adj[node_id]:
                G.add_edge(node_id, adj_id, self.weights[node_id, adj_id])

        G.setup_colors(len(self.input), len(self.hidden), len(self.output))
        G.visualize(show, save)

            
    def get_activations_str(self) -> None:
        vals = [0] * 10
        for id in self.input + self.hidden + self.output:
            vals[self.acts[id]] += 1
        return f'''
--------------Activation Functions--------------------
Linear:  {vals[0]}
Unsigned Step:  {vals[1]}
Sin:  {vals[2]}
Gauss:  {vals[3]}
Tanh:  {vals[4]}
Sigmoid:  {vals[5]}
Inverse:  {vals[6]}
Abs Val:  {vals[7]}
Relu:  {vals[8]}
Cos:  {vals[9]}
------------------------------------------------------'''

    def get_weights_str(self) -> None:
        return f'''
---------------------Weights--------------------------
{os.linesep.join([f"{_from} to {_to}: {weight:.2f}" for (_from, _to), weight in self.weights.items()])}     
------------------------------------------------------'''
    
    def get_bias_str(self) -> None:
        return f'''
----------------------Bias----------------------------
{os.linesep.join([f"{id} has bias: {bias:.2f}" for id, bias in self.bias.items()])}
------------------------------------------------------'''

