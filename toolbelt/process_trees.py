import os
import json
import hashlib
import numpy as np
from .trees import Tree, Node
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
    def __init__(self, name, operating_system=None, ip='x.x.x.x', env=None):
        self.host_id = self.make_id()
        self.name = name
        self.os = operating_system
        self.ip = ip
        self.env = env
        self.process_trees = {}

    def __repr__(self):
        return f"<Instance of Host with ID:{self.host_id}>"

    def __str__(self):
        return f"{self.name} @ {self.ip}"

    @staticmethod
    def make_id():
        return hashlib.md5(str(np.random.rand()).encode('utf8')).hexdigest()[:7]

    def add_tree(self, new_tree):
        """
        Add a new tree to the Host's dict of trees.
        :param new_tree: the tree to add to the host's list of trees
        """
        self.process_trees[new_tree.tree_id] = new_tree

    def to_dict(self):
        ret_val = {'host_id': self.host_id, 'host_name': self.name, 'op_sys': self.os, 'ip': self.ip, 'env': self.env}
        trees_dict = dict()
        for tree_id, tree in self.process_trees.items():
            trees_dict[tree_id] = tree.to_dict()
        ret_val['trees'] = trees_dict
        return ret_val

    def to_json(self, filepath=None, indent=4):
        if not filepath:
            return json.dumps(self.to_dict(), indent=indent)
        else:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)


class ProcessTree(Tree):
    """
    A Tree is a set of processes that have a single, common starting process from which all other processes were spawned
    It can contain only 1 trunk/origin process, but n number of leafs on k branches
    """
    def __init__(self, tree_id=None, host=None):
        Tree.__init__(self, tree_id=tree_id)
        # Inherited =
        #   self.tree_id = self.make_id()
        #   self.nodes = dict()  # {node_id: level }
        #   self.starting_node = None
        #   self.max_width = None
        #   self.max_depth = None
        self.Host = host
        if host:
            self.Host.add_tree(self)

    def __repr__(self):
        return f"<Instance of ProcessTree with ID:{self.tree_id}>"
        # f"<Process Tree {self.tree_num} for {self.Host.name}. with {len(self.nodes.keys())} processes>"

    def __str__(self):
        return f"Process Tree for host {self.Host} with {len(self.nodes.keys())} nodes."

    def to_dict(self):
        ret_val = {'tree_id': self.tree_id}
        if self.Host:
            ret_val['host'] = self.Host.name
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

    def append_proc(self, guid, proc_name, parent_guid=None, timestamp=None, proc_path=None, ignore_structure=False):
        """
        Adds a new node at end of the tree; assumes it is the master node or already has a parent
        :param guid: GUID of the node to add
        :param proc_name: process name for the node we're adding
        :param timestamp: a valid date-time object of when the process was created
        :param proc_path: the full text of the path to the file
        :param parent_guid: GUID of the parent process already in the tree
        :param ignore_structure: if True it will skip checking the tree's structure once the node is added
        """
        # make sure the node isn't in the dict of nodes, and add it
        if guid not in self.nodes.keys():
            new_node = Process(guid=guid, proc_name=proc_name, tree=self, timestamp=timestamp, proc_path=proc_path)
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

    def push_proc(self, guid, proc_name, children_guids=(), timestamp=None, proc_path=None, ignore_structure=False):
        """
        Add a new parent node to the structure, needs to be setting the master node as a child, otherwise it will break
        the tree structure and trip an error
        :param guid: GUID of new node
        :param proc_name: process name for the new node
        :param timestamp: a valid date-time object of when the process was created
        :param proc_path: the full text of the path to the file
        :param children_guids: child node(s) to link to.  Needs to contain the master node...
        :param ignore_structure: if True it will skip checking the tree's structure once the node is added
        """
        if guid not in self.nodes.keys():
            new_node = Process(guid=guid, proc_name=proc_name, tree=self, timestamp=timestamp, proc_path=proc_path)
            self.nodes[guid] = new_node
        else:
            raise ValueError('That node already exists')

        if len(children_guids) > 0:
            for child_guid in children_guids:
                new_node.find_and_add_child(child_guid)
        # Make sure that the node we added did not break the tree structure into multiple trees
        if not ignore_structure and len(self.nodes.keys()) > 1:
            self.check_tree()


class Process(Node):
    """
    A node is a single instance of a process.  It can have only 1 parent (not required) but n children.
    """
    def __init__(self, guid, proc_name, tree, timestamp=None, proc_path=None):
        Node.__init__(self, node_id=guid, name=proc_name, tree=tree,)
        # Inherited =
        #   self.node_id = str(node_id)
        #   self.name = str(name)
        #   self.Tree = tree
        #   self.parent = None
        #   self.children = []
        self.proc_path = str(proc_path)
        self.timestamp = self.fix_date(timestamp)

    def __repr__(self):
        return f"<Instance of Process with ID:{self.node_id} in ProcessTree:{self.Tree.tree_id}>"

    def __str__(self):
        return f"Process Instance of {self.name} with ID {self.node_id} in Process Tree {self.Tree.tree_id}"

    @staticmethod
    def fix_date(passed_date):
        """
        Uses Numpy datetime64 declaration to validate and standardize dates.  If it throws an error for numpy, it will
        here too - I only added handling for null values to keep as None rather than 'NaT'
        :param passed_date: the date from input
        :return: a np.datetime64 time or None
        """
        clean_date = np.datetime64(passed_date)
        if not np.isnat(clean_date):
            return np.datetime64(passed_date)
        else:
            return None

    def to_dict(self):
        ret_val = Node.to_dict(self)
        ret_val['proc_path'] = self.proc_path
        ret_val['timestamp'] = str(self.timestamp)
        return ret_val

    def to_json(self, filepath=None, indent=4):
        if not filepath:
            return json.dumps(self.to_dict(), indent=indent)
        else:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)

    def get_time(self):
        return self.timestamp

    def set_time(self, timestamp):
        self.timestamp = self.fix_date(timestamp)

    def get_path(self):
        return self.proc_path

    def set_path(self, path: str):
        self.proc_path = path


def read_tree(filepath=None, json_str=None, data_dict=None, host=None):
    if filepath:
        data = json.load(open(filepath))
    elif json_str:
        data = json.loads(json_str)
    elif data_dict:
        if isinstance(data_dict, dict):
            data = data_dict
    else:
        raise ValueError('No valid data provided.')

    new_tree = ProcessTree(tree_id=data['tree_id'], host=host)
    num_layers = len(data['nodes'].keys())
    for layer in range(num_layers):
        layer_data = data['nodes'][str(layer)]
        for _, node in layer_data.items():
            parent_id = path = time = None
            if 'parent' in node.keys():
                parent_id = node['parent']
            if 'proc_path' in node.keys():
                path = node['proc_path']
            if 'timestamp' in node.keys():
                time = node['timestamp']
            new_tree.append_proc(guid=node['node_id'],
                                 proc_name=node['name'],
                                 parent_guid=parent_id,
                                 timestamp=time,
                                 proc_path=path)
    return new_tree


def read_host(filepath=None, json_str=None):
    if filepath:
        data = json.load(open(filepath))
    elif json_str:
        data = json.loads(json_str)
    else:
        raise ValueError('No valid data provided.')

    new_host = Host(name=data['host_id'],
                    operating_system=data['op_sys'],
                    ip=data['ip'],
                    env=data['env'])
    for _, tree in data['trees'].items():
        read_tree(data_dict=tree, host=new_host)
    return new_host
