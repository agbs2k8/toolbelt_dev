import numpy as np
import hashlib
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
                max_x = self.max_depth * 3
                max_y = self.max_width * 3
            else:
                max_x, max_y = figsize
        else:
            pos = self.make_layout(horizontal=False)
            if not figsize:
                max_y = self.max_depth * 3
                max_x = self.max_width * 3
            else:
                max_x, max_y = figsize

        font_size = int(max_x * (4 / 5))
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
        ax.set_title(f'Plot of Tree: {self.tree_id}')
        if save_path:
            plt.savefig(save_path)
        return ax


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
