#!python
#cython: language_level=3
import json
import copy
import hashlib
import numpy as np
from .trees import Tree, Node
"""
Notes on how to import from the github tracked directory:
    sys.path.append('/Users/ajwilson/GitRepos/toolbelt_dev/')
    from toolbelt.process_tress import Host, Tree, Node
"""


class Master:
    def __init__(self, name, max_magnitude=7):
        self.name=name
        self.master_trees = dict()  # {master_tree_id: tree_obj, ...}
        self.tree_host_mapping = dict()  # {master_tree_id: {"host_name": "tree_id_from_host.unique_trees", ...}, ...}
        self.magnitude = max_magnitude  # the number of available digits for the unique tree IDs

    def __repr__(self):
        return f"<Master for: {self.name}>"

    def __str__(self):
        return f"{self.name}"

    def set_tree_host_mapping(self, data):
        self.tree_host_mapping = data

    def get_tree_host_mapping(self):
        return self.tree_host_mapping

    def to_dict(self):
        ret_val = {'name': self.name,
                   'tree_host_mapping': self.tree_host_mapping}
        trees_dict = dict()
        for tree_id, tree in self.master_trees.items():
            trees_dict[tree_id] = tree.to_dict()
        ret_val['master_trees'] = trees_dict
        return ret_val

    def to_json(self, filepath=None, indent=4):
        if not filepath:
            return json.dumps(self.to_dict(), indent=indent)
        else:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)

    def init_tree_id(self):
        return f"{self.name}_MasterTree_{str(1).zfill(self.magnitude)}"

    def next_tree_id(self):
        return f"{self.name}_MasterTree_" \
            f"{str(int(max(list(self.master_trees.keys())).split('_')[-1])+1).zfill(self.magnitude)}"

    def make_master_copy(self, new_tree, master_tree_id):
        master_copy = new_tree.copy()
        master_copy.set_id(master_tree_id)
        master_copy.force_new_host(self)
        return master_copy

    def add_tree(self, new_tree, reading_data=True):
        self.master_trees[new_tree.tree_id] = new_tree

    def include_tree(self, new_tree):
        original_host_name = new_tree.Host.name
        original_host_tree_guid = new_tree.tree_id
        # Make sure this isn't the first one we try to add...
        if len(self.master_trees.keys()) > 0:  # ... if this isn't the first...
            # I need to check and see if a matching tree is already in the master
            for existing_master_id, existing_master_tree in self.master_trees.items():  # go through all the existing
                if existing_master_tree.matches(new_tree):  # IF I FIND A MATCH!!!
                    # THAT ONE MATCHED - dont add to master_trees, but update tree_host_mapping
                    # the host of the new tree should NOT already be in the tree_host_mapping:
                    if original_host_name not in self.tree_host_mapping[existing_master_id].keys():
                        #Update
                        self.tree_host_mapping[existing_master_id][original_host_name] = original_host_tree_guid
                    else:
                        raise KeyError('Duplicated in the host mapping - duplicate trees from same host uploaded to '
                                       'master')
                    return  # I'm done there, dont need to check the others...
                else:
                    # THAT ONE DIDN'T MATCH
                    continue
            # NONE OF THEM MATCHED
            master_tree_id = self.next_tree_id()
            self.master_trees[master_tree_id] = self.make_master_copy(new_tree, master_tree_id)
            self.tree_host_mapping[master_tree_id] = {original_host_name: original_host_tree_guid}
            return
        else:  # ...if this is the first one we try to add...
            master_tree_id = self.init_tree_id()
            self.master_trees[master_tree_id] = self.make_master_copy(new_tree, master_tree_id)
            self.tree_host_mapping[master_tree_id] = {original_host_name: original_host_tree_guid}
            return

    def containing_hosts(self, master_tree_id):
        return list(self.tree_host_mapping[master_tree_id].keys())

    def find_in_master(self, new_tree):
        for master_tree_id, master_tree in self.master_trees.items():
            if master_tree.matches(new_tree):
                return master_tree_id
        return None


class Host:
    """
    Hosts are the individual computers that I want to create 1 or more process trees for.  They store common information
    about the machine and a list of the trees contained within it.
    """
    def __init__(self, name, operating_system=None, ip='x.x.x.x', env=None, host_id=None):
        if host_id:
            self.host_id = host_id
        else:
            self.host_id = self.make_id()
        self.name = name
        self.os = operating_system
        self.ip = ip
        self.env = env
        self.process_trees = dict()  # {tree_id: tree_obj...}
        # self.unique_trees = dict()  # {tree_id: [matching_trees,...], ...}
        self.tree_iterations = dict()  # {tree_id: [(start_time, elapsed, start_node_GUID), ...], ...}

    def __repr__(self):
        return f"<Instance of Host with ID:{self.host_id}>"

    def __str__(self):
        return f"{self.name} @ {self.ip}"

    @staticmethod
    def make_id():
        return hashlib.md5(str(np.random.rand()).encode('utf8')).hexdigest()[:7]

    def get_trees(self):
        return [tree for _, tree in self.process_trees.items()]

    def has_tree(self, new_tree):
        for tree in self.get_trees():
            if tree.matches(new_tree):
                return True
        return False

    def set_tree_iterations(self, tree_iters):
        self.tree_iterations = tree_iters

    def get_tree_iterations(self):
        return self.tree_iterations

    def find_match(self, new_tree):
        results = [x.tree_id for x in filter(lambda x: x.matches(new_tree), self.get_trees())]
        if len(results) > 0:
            return results[0]
        else:
            return None

    def add_tree(self, new_tree, reading_data=True):
        if not reading_data:
            existing_id = self.find_match(new_tree)
            if existing_id:
                self.tree_iterations[existing_id].append(new_tree.get_run_data())
            else:
                self.process_trees[new_tree.tree_id] = new_tree
                self.tree_iterations[new_tree.tree_id] = [new_tree.get_run_data()]
                if new_tree.Host != self:
                    new_tree.force_new_host(self)
        else:
            self.process_trees[new_tree.tree_id] = new_tree

    def to_dict(self):
        ret_val = {'host_id': self.host_id,
                   'host_name': self.name,
                   'op_sys': self.os,
                   'ip': self.ip,
                   'env': self.env,
                   'tree_iterations': self.tree_iterations}
        trees_dict = dict()
        for tree_id, tree in self.process_trees.items():
            trees_dict[tree_id] = tree.to_dict()
        ret_val['process_trees'] = trees_dict
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

    def copy(self):
        return copy.deepcopy(self)

    def get_host(self):
        return Host

    def set_host(self, host):
        self.Host = host
        if host:
            self.Host.add_tree(self)

    def force_new_host(self, host):
        self.Host = host

    def get_elapsed_time(self):
        try:
            elapsed = max([node.timestamp for node in self.get_leafs()]) - self.starting_node.timestamp
            return np.timedelta64(elapsed).item().total_seconds()
        except TypeError:
            return None

    def get_start_time(self):
        return self.starting_node.timestamp

    def get_run_data(self):
        return str(self.get_start_time()), self.get_elapsed_time(), self.starting_node.node_id

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

    def get_proc_name(self):
        return self.name


def read_process_tree(filepath=None, json_str=None, data_dict=None, host=None):
    if filepath:
        data = json.load(open(filepath))
    elif json_str:
        data = json.loads(json_str)
    elif data_dict:
        if isinstance(data_dict, dict):
            data = data_dict
    else:
        raise ValueError('No valid data provided.')
    new_tree = ProcessTree(tree_id=data['tree_id'], host=None)
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
    if host:
        host.add_tree(new_tree, reading_data=True)
        new_tree.force_new_host(host)
    return new_tree


def read_host(filepath=None, json_str=None):
    if filepath:
        data = json.load(open(filepath))
    elif json_str:
        data = json.loads(json_str)
    else:
        raise ValueError('No valid data provided.')

    new_host = Host(name=data['host_name'],
                    operating_system=data['op_sys'],
                    ip=data['ip'],
                    env=data['env'],
                    host_id=data['host_id'])
    new_host.set_tree_iterations(data['tree_iterations'])
    for _, tree in data['process_trees'].items():
        read_process_tree(data_dict=tree, host=new_host)
    return new_host


def read_master(filepath=None, json_str=None):
    if filepath:
        data = json.load(open(filepath))
    elif json_str:
        data = json.loads(json_str)
    else:
        raise ValueError('No valid data provided.')
    new_master = Master(name=data['name'])
    new_master.set_tree_host_mapping(data['tree_host_mapping'])
    for _, tree in data['master_trees'].items():
        read_process_tree(data_dict=tree, host=new_master)
    return new_master


def find_all_matches(host, new_tree):
    results = [x.tree_id for x in filter(lambda x: x.matches(new_tree), host.get_trees())]
    if len(results) > 0:
        return results
    else:
        return None

def build_master_from_hosts(host_list, master_name):
    new_master = Master(name=master_name)
    for host in host_list:
        if isinstance(host, str):
            host = read_host(filepath=host)
        for tree in host.get_trees():
            new_master.include_tree(tree)
    return new_master
