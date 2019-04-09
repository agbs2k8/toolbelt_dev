import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
"""
Notes on how to import from the github tracked directory:
    sys.path.append('/Users/ajwilson/GitRepos/toolbelt_dev/')
    from toolbelt.process_tress import Host, Tree, Node
"""


class Host:
    """
    Hosts are the individual computers that I want to create 1 or more process trees for.  They store common information
    about the machine and a list of the trees contained within it.
    """
    def __init__(self, name):
        self.name = name
        self.os = None
        self.ip = None
        self.env = None
        self.trees = []

    def __repr__(self):
        ret_string = f"<Host Name: {self.name} with the following trees:\n>"
        for tree in self.trees:
            ret_string += f"{tree.__repr__}\n"
        return ret_string

    def add_tree(self, tree):
        """
        Add a new tree to the Host's list of trees.  We're not tracking order, but want the tree to track it's own
        index in the list of the Host's trees
        :param tree: the tree to add to the host's list of trees
        :return: the index of this tree in the list of host's trees
        """
        self.trees.append(tree)
        return self.trees.index(tree)


class Tree:
    """
    A Tree is a set of processes that have a single, common starting process from which all other processes were spawned
    It can contain only 1 trunk/origin process, but n number of leafs on k branches
    """
    def __init__(self, host):
        self.levels = []
        self.Host = host
        self.tree_num = host.add_tree(self)
        self.nodes = dict()  # {node_guid: level }
        self.starting_node = None
        self.max_width = None
        self.max_depth = None

    def __repr__(self):
        return f"<Process Tree {self.tree_num} for {self.Host.name}. with {len(self.nodes.keys())} processes>"

    def to_list(self):
        """
        Condense the tree into a nested list for easy visualization.  NOT HASHABLE because there is not control on order
        :return: nested list of all processes in the tree
        """
        if self.starting_node:
            return self.starting_node.get_branch_to_leaf()
        else:
            self.check_tree()
            if self.starting_node:
                return self.starting_node.get_branch_to_leaf()
            else:
                raise ValueError('Cannot find starting node; check that nodes exists.')

    def matches(self, other):
        """
        Compares two trees (this one to another) and makes sure that they have 1 common starting process (by proc_name)
        *not by GUID*, makes sure they have the exact same set of leafs, and then compares all branches between the
        leafs and the trunk to make sure they match
        :param other: another tree to compare to
        :return: True if the match, False if they are different
        """
        # 0. Make sure we have a starting node:
        self.check_tree()
        if not self.starting_node:
            return False
        # 1. Make sure the trunk/origin process is the same for both
        if self.starting_node.process_name != other.starting_node.process_name:
            return False
        # ...Now we compare the leafs...
        my_leafs = self.get_leafs()
        ot_leafs = other.get_leafs()
        # 2. Make sure the set of leafs processes are the same
        if set([leaf.process_name for leaf in my_leafs]) != set([leaf.process_name for leaf in ot_leafs]):
            return False
        # 3. compare the contents of each branch and make sure there is a matching one in each
        my_branches = [[node.process_name for node in leaf.get_branch_to_trunk()] for leaf in my_leafs]
        ot_branches = [[node.process_name for node in leaf.get_branch_to_trunk()] for leaf in ot_leafs]
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

    def find_node(self, node_guid):
        """
        Lookup Node object by the GUID from the parent's dictionary
        :param node_guid: duh...
        :return: Node
        """
        return self.nodes[node_guid]

    def append_node(self, guid, proc_name, parent_guid=None, ignore_structure=False) -> None:
        """
        Adds a new node at end of the tree; assumes it is the master node or already has a parent
        :param guid: GUID of the node to add
        :param proc_name: process name for the node we're adding
        :param parent_guid: GUID of the parent process already in the tree
        :param ignore_structure: if True it will skip checking the tree's structure once the node is added
        """
        # make sure the node isn't in the dict of nodes, and add it
        if guid not in self.nodes.keys():
            new_node = Node(guid, proc_name, self)
            self.nodes[guid] = new_node
        else:  # if the node already exists, raise an error
            raise ValueError('That node already exists.')

        # if they passed a GUID for a parent, try looking it up and adding it
        if parent_guid and parent_guid in self.nodes.keys():
            new_node.find_and_set_parent(parent_guid)
        # If the parent GUID is invalid
        elif parent_guid and parent_guid not in self.nodes.keys():
            raise ValueError('The designated parent GUID does not exist.')
        # Make sure that the node we added did not break the tree structure into multiple trees
        if not ignore_structure and len(self.nodes.keys()) > 1:
            self.check_tree()

    def push_node(self, guid, proc_name, children_guids=(), ignore_structure=False) -> None:
        """
        Add a new parent node to the structure, needs to be setting the master node as a child, otherwise it will break
        the tree structure and trip an error
        :param guid: GUID of new node
        :param proc_name: process name for the new node
        :param children_guids: child node(s) to link to.  Needs to contain the master node...
        :param ignore_structure: if True it will skip checking the tree's structure once the node is added
        """
        if guid not in self.nodes.keys():
            new_node = Node(guid, proc_name, self)
            self.nodes[guid] = new_node
        else:
            raise ValueError('That node already exists')

        if len(children_guids) > 0:
            for child_guid in children_guids:
                new_node.find_and_add_child(child_guid)
        # Make sure that the node we added did not break the tree structure into multiple trees
        if not ignore_structure and len(self.nodes.keys()) > 1:
            self.check_tree()

    def check_tree(self) -> None:
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
        for guid, node in self.nodes.items():
            if not trunk:
                trunk = node.get_trunk_node()
            else:
                is_valid = (trunk == node.get_trunk_node())
        if not is_valid:
            raise ValueError('Incorrect tree structure.')
        self.starting_node = trunk

    def get_leafs(self):
        """
        Find all leaf nodes
        :return: A list of the node objects that have no children
        """
        return [node for guid, node in self.nodes.items() if len(node.get_children()) < 1]

    def get_node_list(self):
        """
        Get the list of nodes out of the dict that contains them
        :return: a list of all nodes
        """
        return [node for guid, node in self.nodes.items()]

    def get_edge_list(self):
        """
        For building the Graph... GUID -> GUID mapped edges
        :return: tuple of (from, to) GUIDs of nodes in tree
        """
        edges = []
        for node in self.get_node_list():
            if node.get_parent():
                edges.append((node.get_parent().guid, node.guid))
        return edges

    def get_depth(self):
        """
        Find the distance from the starting_node to the furthest leaf node.  For building visualizations so we know
        how many layers are needed.  Also set's the variable for the tree
        :return: integer depth from trunk/origin to furthest leaf
        """
        # 0. Make sure we have a starting node:
        self.check_tree()
        if not self.starting_node:
            return False
        # 1. Find max depth from staring node
        self.max_depth = self.starting_node.subtree_depth()
        return self.max_depth

    def to_graph(self):
        """
        Create NetworkX Directed Graph of the tree.  Nodes tracked by GUID
        :return: NetworX DiGraph obj
        """
        g = nx.DiGraph()
        g.add_nodes_from([node.guid for node in self.get_node_list()])
        g.add_edges_from(self.get_edge_list())
        return g

    def make_layout(self, horizontal=True):
        """
        Map all tree nodes to (x,y) coordinates where x & y are each in range [0,1] so they can be plotted
        :param horizontal: by default it plots left/right, if False it flips to plot up/down
        :return: dict of {node_guid: [x,y], ...}
        """
        self.max_depth = self.get_depth()
        leafs = self.get_leafs()
        self.max_width = len(leafs)
        x_options = np.linspace(0, 1, self.max_depth)
        y_options = np.linspace(0, 1, self.max_width)
        pos = {self.starting_node.guid: [x_options[0], None]}
        layers = [[self.starting_node]]
        for i in range(self.max_depth - 1):
            next_layer = []
            for node in layers[i]:
                next_layer += node.get_children()
            for node in next_layer:
                pos[node.guid] = [x_options[i + 1], None]
            layers.append(next_layer)
        for i, leaf in enumerate(leafs):
            pos[leaf.guid][1] = y_options[i]
            parent = leaf.get_parent()
            while parent:
                pos[parent.guid][1] = y_options[i]
                parent = parent.get_parent()
        if horizontal:
            return {key: np.array(val, dtype=float) for key, val in pos.items()}
        else:
            return {key: np.array([1, 0])-np.array(val[::-1], dtype=float) for key, val in pos.items()}

    def label_dict(self):
        """
        For making plots - maps GUID to process name
        :return: dict {guid: process_name, ...}
        """
        return {guid: node.process_name for guid, node in self.nodes.items()}

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
        ax.set_title(f'Process Flow for Tree Number: {self.tree_num} of Host {self.Host.name}')
        if save_path:
            plt.savefig(save_path)
        return ax


class Node:
    def __init__(self, guid, proc_name, tree):
        self.guid = guid
        self.parent = None
        self.children = []
        self.process_name = proc_name
        self.Tree = tree

    def __repr__(self):
        return f"<Instance of the process '{self.process_name}' with GUID: '{self.guid}' in '{self.Tree.Host.name}' " \
            f"tree '{self.Tree.tree_num}'.>"

    def __eq__(self, other):
        if self.process_name == other.process_name:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.guid)

    def find_and_set_parent(self, parent_guid):
        self.parent = self.Tree.find_node(parent_guid)
        self.parent.add_child(self)

    def find_and_add_child(self, child_guid):
        child_node = self.Tree.find_node(child_guid)
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
        :return:
        """
        parent = self.get_parent()
        if parent:
            return parent.get_trunk_node()
        else:
            return self

    def get_last_leaf(self):
        """
        Recursively find the last leaf node of the tree
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
            return [self.process_name, [child.get_branch_to_leaf() for child in self.children]]
        else:
            return [self.process_name]

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
        if self.guid != self.Tree.starting_node.guid:
            branch.append(self.Tree.starting_node)
        return branch

    def same_children(self, other) -> bool:
        """
        Make sure that two process nodes have the same children processes
        :param other: another node
        :return: True if the children process names match
        """
        if len(self.children) > 0 and len(self.children) == len(other.children):
            return set([child.process_name for child in self.iter_child()]) == set(
                [other.process_name for other in other.iter_child()])
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
