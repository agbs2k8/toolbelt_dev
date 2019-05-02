#!python
#cython: language_level=3
import json
import hashlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Tree:
    """
    A Tree is a set of nodes that have a single, common trunk/origin/starting node from which all other branch out
    It can contain only 1 trunk/origin node, but n number of leafs on k branches
    """
    def __init__(self, tree_id=None, *args, **kwargs):
        if tree_id:
            self.tree_id = str(tree_id)
        else:
            self.tree_id = self.make_id()
        self.nodes = dict()  # {node_id: level }
        self.starting_node = None
        self.max_width = None
        self.max_depth = None

    def __repr__(self):
        return f"<Instance of Tree with ID:{self.tree_id}>"

    def __str__(self):
        return f"Tree ID '{self.tree_id}' with {len(self.nodes.keys())} Nodes."

    @staticmethod
    def make_id():
        return hashlib.md5(str(np.random.rand()).encode('utf8')).hexdigest()[:10]

    def set_id(self, new_id):
        self.tree_id = new_id

    def to_list(self):
        """
        Condense the tree into a nested list for easy visualization.  NOT HASHABLE because there is not control on order
        :return: nested list of all nodes in the tree
        """
        if self.starting_node:
            return self.starting_node.get_branch_to_leaf()
        else:
            self.check_tree()
            if self.starting_node:
                return self.starting_node.get_branch_to_leaf()
            else:
                raise ValueError('Cannot find starting node; check that nodes exists.')

    def print_tree(self, return_string=False):
        self.check_tree()
        result = self.starting_node.print_branch()
        if return_string:
            return result
        else:
            print(result)

    def get_layers(self):
        num_layers = self.get_depth()
        layers = [[self.starting_node]]
        for i in range(num_layers-1):
            new_layer = []
            for node in layers[i]:
                new_layer += node.get_children()
            layers.append(new_layer)
        return layers

    def to_dict(self):
        ret_val = {'tree_id': self.tree_id}
        layers = self.get_layers()
        layers_dict = dict()
        for layer_idx, layer in enumerate(layers):
            layer_dict = dict()
            for node_idx, node in enumerate(layer):
                layer_dict[node_idx] = node.to_dict()
            layers_dict[layer_idx] = layer_dict
        ret_val['nodes'] = layers_dict
        return ret_val

    def to_json(self, filepath=None, indent=4):
        if not filepath:
            return json.dumps(self.to_dict(), indent=indent)
        else:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)

    def find_node(self, node_id):
        """
        Lookup Node object by the node_id from the parent's dictionary
        :param node_id: duh...
        :return: Node
        """
        return self.nodes[node_id]

    def append_node(self, node_id, name, parent_id=None, ignore_structure=False):
        """
        Adds a new node at end of a branch; assumes it is the trunk/origin node or already has a parent
        :param node_id: id of new node
        :param name: name of new node
        :param parent_id: id of the preceding node in the branch - if none it must be the trunk/origin node
        :param ignore_structure: False to skip error checking - can cause errors later
        :return: Nothing
        """
        # make sure the node isn't in the dict of nodes, and add it
        if node_id not in self.nodes.keys():
            new_node = Node(node_id=node_id, name=name, tree=self)
            self.nodes[node_id] = new_node
        else:  # if the node already exists, raise an error
            raise ValueError('That node id already exists.')

        # if they passed a id for a parent, try looking it up and adding it
        if parent_id and parent_id in self.nodes.keys():
            new_node.find_and_set_parent(parent_id)
        # If the parent node_id is invalid
        elif parent_id and parent_id not in self.nodes.keys():
            raise ValueError('The designated parent ID does not exist.')
        # Make sure that the node we added did not break the tree structure into multiple trees
        if not ignore_structure and len(self.nodes.keys()) > 1:
            self.check_tree()

    def push_node(self, node_id, name, children_ids=(), ignore_structure=False):
        """
        Add a new parent node to the structure, needs to be setting the master node as a child, otherwise it will break
        the tree structure and trip an error (unless you force it to ignore that)
        :param node_id: id of new node
        :param name: name of new node
        :param children_ids: ids of any child node(s) already in the tree
        :param ignore_structure: False to skip the error checking - can cause failures later
        :return: Nothing
        """
        if node_id not in self.nodes.keys():
            new_node = Node(node_id=node_id, name=name, tree=self)
            self.nodes[node_id] = new_node
        else:
            raise ValueError('That node already exists')

        if len(children_ids) > 0:
            for child_id in children_ids:
                new_node.find_and_add_child(child_id)
        # Make sure that the node we added did not break the tree structure into multiple trees
        if not ignore_structure and len(self.nodes.keys()) > 1:
            self.check_tree()

    def delete_node(self, node):
        if not isinstance(node, Node) and isinstance(node, str):
            node = self.find_node(node)
        parent = node.get_parent()
        if parent:
            parent.children.remove(node)
        if len(node.get_children()) > 0:
            for child in node.iter_child():
                child.set_parent(None)
        self.nodes = {_id: _node for _id, _node in self.nodes.items() if _id != node.node_id}
        self.check_tree()

    def check_tree(self):
        """
        Ensure that we still have a valid tree structure with only 1 trunk/origin node shared by all other nodes
        Also, set the tree's starting_node value, once validated, in case it is not yet defined or has changed
        """
        trunk = None
        is_valid = False
        # catch any issues if this is called when the tree is empty or a single node
        if len(self.nodes.keys()) == 0:
            raise KeyError('The tree is empty.')
        elif len(self.nodes.keys()) == 1:
            self.starting_node = self.nodes[list(self.nodes.keys())[0]]
            return
        # assuming there is more than a single node, make sure it is a single tree
        for _, node in self.nodes.items():
            if not trunk:
                trunk = node.get_trunk_node()
            else:
                is_valid = (trunk == node.get_trunk_node())
        if not is_valid:
            raise ValueError('Incorrect tree structure.')
        self.starting_node = trunk

    def get_leafs(self):
        """
        Find all leaf nodes = node objects that have no children
        :return: list of leaf node objects
        """
        return [node for _, node in self.nodes.items() if len(node.get_children()) < 1]

    def get_node_list(self):
        """
        Get the list of nodes out of the dict that contains them
        :return: a list of all node objects
        """
        return [node for _, node in self.nodes.items()]

    def get_edge_list(self):
        """
        For building the Graph... node_id -> node_id mapped /directional edges
        :return: list of tuples of (from, to) node_id of nodes in tree
        """
        edges = []
        for node in self.get_node_list():
            if node.get_parent():
                edges.append((node.get_parent().node_id, node.node_id))
        return edges

    def get_depth(self):
        """
        Find the distance from the starting_node to the furthest leaf node.  For building visualizations so we know
        how many layers are needed.  Also set's the variable for the tree's starting node if not already identified
        :return: integer depth from trunk/origin to furthest leaf
        """
        self.check_tree()
        if not self.starting_node:
            raise KeyError('Starting node not found.')
        self.max_depth = self.starting_node.subtree_depth()
        return self.max_depth

    def get_width(self):
        self.max_width = len(self.get_leafs)
        return self.max_width

    def to_graph(self):
        """
        Create NetworkX Directed Graph of the tree.  Nodes tracked by node_id
        :return: NetworkX DiGraph obj
        """
        g = nx.DiGraph()
        g.add_nodes_from([node.node_id for node in self.get_node_list()])
        g.add_edges_from(self.get_edge_list())
        return g

    def make_layout(self, horizontal=True):
        """
        Map all tree nodes to (x,y) coordinates where x & y are each in range [0,1] so they can be plotted
        :param horizontal: by default it plots left/right, if False it flips to plot up/down
        :return: dict of {node_id: [x,y], ...}
        """
        self.max_depth = self.get_depth()
        leafs = self.get_leafs()
        self.max_width = len(leafs)
        x_options = np.linspace(0, 1, self.max_depth)
        y_options = np.linspace(0, 1, self.max_width)
        pos = {self.starting_node.node_id: [x_options[0], None]}
        layers = [[self.starting_node]]
        for i in range(self.max_depth - 1):
            next_layer = []
            for node in layers[i]:
                next_layer += node.get_children()
            for node in next_layer:
                pos[node.node_id] = [x_options[i + 1], None]
            layers.append(next_layer)
        for i, leaf in enumerate(leafs):
            pos[leaf.node_id][1] = y_options[i]
            parent = leaf.get_parent()
            while parent:
                pos[parent.node_id][1] = y_options[i]
                parent = parent.get_parent()
        if horizontal:
            return {key: np.array(val, dtype=float) for key, val in pos.items()}
        else:
            return {key: np.array([1, 0])-np.array(val[::-1], dtype=float) for key, val in pos.items()}

    def label_dict(self):
        """
        For making plots - maps node_id to node name
        :return: dict {node_id: name, ...}
        """
        return {node_id: node.name for node_id, node in self.nodes.items()}

    @staticmethod
    def fix_plot_ratios(max_x, max_y):
        ratio = max_x / max_y
        if ratio < 1:
            return max_x, max_x
        elif ratio > (5/3):
            return max_x, (3/5)*max_x
        else:
            return max_x, max_y

    def plot(self, figsize=None, save_path=None, horizontal=True):
        """
        Create Matplotlib Figure of the graph
        :param figsize: (x, y) values of the size of the plot.  By default is set based on the height/width of graph
        :param save_path: if a path is provided, the graph will be saved to disk
        :param horizontal: by default it plots left/right, if False it flips to plot up/down
        :return: matplotlib plot object
        """
        g = self.to_graph()
        if horizontal:
            pos = self.make_layout()
            if not figsize:
                max_x, max_y = self.fix_plot_ratios(self.max_depth*4, self.max_width*4)
            else:
                max_x, max_y = figsize
        else:
            pos = self.make_layout(horizontal=False)
            if not figsize:
                max_y, max_x = self.fix_plot_ratios(self.max_depth*4, self.max_width*4)
            else:
                max_x, max_y = figsize

        font_size = int(max_x)
        node_size = max_x * ((2 / 3) * 1000)

        fig, ax = plt.subplots(figsize=(max_x, max_y))

        nx.draw_networkx_nodes(g, pos,
                               node_color='lightgray',
                               node_size=node_size)
        nx.draw_networkx_edges(g, pos,
                               node_size=node_size,
                               arrowsize=max_x)
        nx.draw_networkx_labels(g, pos,
                                self.label_dict(),
                                font_size=font_size)
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        return ax

    def matches(self, other):
        """
        Compares two trees (this one to another) and makes sure that they have 1 common starting process (by name)
        *not by ID*, makes sure they have the exact same set of leafs, and then compares all branches between the
        leafs and the trunk to make sure they match
        :param other: another tree to compare to
        :return: True if the match, False if they are different
        """
        # TODO: Look more into Graph Isomorphism algorithms

        # 0. Make sure we have a starting node:
        self.check_tree()
        if not self.starting_node:
            return False
        # 1. Make sure the trunk/origin process is the same for both
        if self.starting_node.name != other.starting_node.name:
            return False
        # ...Now we compare the leafs...
        my_leafs = self.get_leafs()
        ot_leafs = other.get_leafs()
        # 2. Make sure the set of leafs processes are the same
        if set([leaf.name for leaf in my_leafs]) != set([leaf.name for leaf in ot_leafs]):
            return False
        # 3. compare the contents of each branch and make sure there is a matching one in each
        my_branches = [[node.name for node in leaf.get_branch_to_trunk()] for leaf in my_leafs]
        ot_branches = [[node.name for node in leaf.get_branch_to_trunk()] for leaf in ot_leafs]
        while len(my_branches) > 0:
            cur_branch = my_branches.pop()
            if cur_branch in ot_branches:
                ot_branches.remove(cur_branch)
            else:
                return False
        # ... make sure there wasn't anything else left in the the other one.
        if len(ot_branches) == 0:
            return True
        else:
            return False

    def subtree(self, starting_node):
        """
        Returns a duplicated tree of all branches from the given node
        :param starting_node: either Node OBJ or node_id to look up and retrieve the node
        :return: Tree
        """
        if isinstance(starting_node, str):
            starting_node = self.find_node(starting_node)
        new_tree = Tree()
        starting_node.copy_to(new_tree, with_parent=False)
        children = starting_node.get_children()
        for _ in range(starting_node.subtree_depth()-1):
            next_layer = []
            for child in children:
                child.copy_to(new_tree)
                next_layer += child.get_children()
            children = next_layer
        return new_tree

    def has_subtree(self, other):
        first_proc_name = other.starting_node.name
        prospects = [sub_start_node for _, sub_start_node in self.nodes.items() if
                     sub_start_node.name == first_proc_name]
        search_depth = other.get_depth()
        # The other cannot be larger than I am
        if self.get_depth() < search_depth:
            return False
        # Make sure the passed set is a subset of what I already have
        elif not set([node.name for _, node in other.nodes.items()]).issubset(
                set([node.name for _, node in self.nodes.items()])):
            return False
        # Make sure I have some processes that match the starting process name
        elif len(prospects) < 1:
            return False
        # Now go through all of the prospective starting points...
        else:
            for prospect in prospects:
                prospect_matched = True
                pr_children = [x for x in prospect.get_children()]
                ot_children = [x for x in other.starting_node.get_children()]
                for i in range(search_depth - 1):
                    # fill these for the next iteration as we go
                    pr_next_layer = []
                    ot_next_layer = []
                    for ot_child in ot_children:
                        ot_next_layer += ot_child.get_children()
                        child_matched = False
                        for pr_child in pr_children:
                            if ot_child.name == pr_child.name and ot_child.parent.name == pr_child.parent.name:
                                pr_children.remove(pr_child)
                                pr_next_layer += pr_child.get_children()
                                child_matched = True
                        if not child_matched:
                            prospect_matched = False
                    pr_children = pr_next_layer
                    ot_children = ot_next_layer
                if prospect_matched:
                    return True
        return False

    def is_subtree(self, other):
        return other.has_subtree(self)


class Node:
    """
    A node is a single step in the tree. It can be mind branch, a split in a branch, or a leaf
    It can have only 1 parent (not required if it is the trunk/origin/start) but n children.
    There can be nodes with duplicate names which are equal to one another but have different IDs
    """
    def __init__(self, node_id, name, tree):
        self.node_id = str(node_id)
        self.name = str(name)
        self.Tree = tree
        self.parent = None
        self.children = []

    def __repr__(self):
        return f"<Instance of Node with ID:{self.node_id}>"

    def __str__(self):
        return f"Node ID: '{self.node_id}', Name: '{self.name}' on TreeID: '{self.Tree.tree_id}'."

    def __eq__(self, other):
        if self.name == other.name:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.name)

    def copy_to(self, new_tree, with_parent=True):
        if with_parent:
            new_tree.append_node(node_id=self.node_id, name=self.name, parent_id=self.parent.node_id)
        else:
            new_tree.append_node(node_id=self.node_id, name=self.name, parent_id=None)

    def to_dict(self):
        ret_val = {'node_id': self.node_id, 'name': self.name, 'tree_id': self.Tree.tree_id}
        if self.parent:
            ret_val['parent'] = self.parent.node_id
        if len(self.children) > 0:
            child_dict = dict()
            for idx, child in enumerate(self.iter_child()):
                child_dict[idx] = child.node_id
            ret_val['children'] = child_dict
        return ret_val

    def to_json(self, filepath=None, indent=4):
        if not filepath:
            return json.dumps(self.to_dict(), indent=indent)
        else:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)

    def find_and_set_parent(self, parent_id):
        self.parent = self.Tree.find_node(parent_id)
        self.parent.add_child(self)

    def find_and_add_child(self, child_id):
        child_node = self.Tree.find_node(child_id)
        self.children.append(child_node)
        if not child_node.parent:
            child_node.set_parent(self)

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def add_child(self, child):
        self.children.append(child)

    def iter_child(self):
        for child in self.children:
            yield child

    def get_trunk_node(self):
        """
        Recursively find the origin/parent of the tree
        :return: trunk/origin/starting node
        """
        parent = self.get_parent()
        if parent:
            return parent.get_trunk_node()
        else:
            return self

    def get_last_leaf(self):
        """
        Recursively find the last leaf node of the sub-tree sprouting from this node
        :return: leaf furthest from the trunk/origin
        """
        children = self.get_children()
        if children:
            for child in children:
                return child.get_last_leaf()
        else:
            return self

    def get_branch_to_leaf(self):
        """
        Node method for creating the "to_list" tree method. Recursively search to all leafs out from the node
        :return: list of subsequent node branches
        """
        if len(self.children) > 0:
            return [self.name, [child.get_branch_to_leaf() for child in self.children]]
        else:
            return [self.name]

    def get_branch_to_trunk(self):
        """
        Create list of nodes from a given node back to the trunk/origin
        :return:
        """
        branch = [self]
        parent = self.get_parent()
        while parent and parent != self.Tree.starting_node:
            branch.append(parent)
            parent = parent.get_parent()
        if self.node_id != self.Tree.starting_node.node_id:
            branch.append(self.Tree.starting_node)
        return branch

    def same_children(self, other) -> bool:
        """
        Make sure that two nodes have the same children nodes by name
        :param other: another node
        :return: True if the children names match
        """
        if len(self.children) > 0 and len(self.children) == len(other.children):
            return set([child.name for child in self.iter_child()]) == set(
                [other.name for other in other.iter_child()])
        elif len(self.children) == 0 and len(other.childrent) == 0:
            return True
        else:
            return False

    def subtree_depth(self):
        """
        Recursively find the max depth of the last leaf node branched out from our node
        :return: max depth from the selected node
        """
        if self is None:
            return 0
        else:
            max_depth = 0
            for child in self.get_children():
                depth = child.subtree_depth()
                if depth > max_depth:
                    max_depth = depth
            return max_depth + 1

    def print_branch(self):
        depth = 4*(len(self.get_branch_to_trunk())-1)
        if self.parent and len(self.children) > 0:
            return (" "*(depth-4))+'└── '+self.name+"\n"+"".join([child.print_branch() for child in self.children])
        elif len(self.children) > 0:
            return (" "*(depth-4))+self.name+"\n"+"".join([child.print_branch() for child in self.children])
        else:
            return (" "*(depth-4))+'└── '+self.name+"\n"


def read_tree(filepath=None, json_str=None, data_dict=None):
    if filepath:
        data = json.load(open(filepath))
    elif json_str:
        data = json.loads(json_str)
    elif data_dict:
        if isinstance(data_dict, dict):
            data = data_dict
    else:
        raise ValueError('No valid data provided.')
    new_tree = Tree(tree_id=data['tree_id'])
    num_layers = len(data['nodes'].keys())
    for layer in range(num_layers):
        layer_data = data['nodes'][str(layer)]
        for _, node in layer_data.items():
            parent_id  = None
            if 'parent' in node.keys():
                parent_id = node['parent']
            new_tree.append_node(node_id=node['node_id'],
                                 name=node['name'],
                                 parent_id=parent_id)
    return new_tree
