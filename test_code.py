import datetime
import multiprocessing
from toolbelt.process_trees import read_host
from toolbelt.utils import batch


def find_all_matches(host, new_tree, n_jobs=None):
    result = []
    pool = multiprocessing.Pool()
    if not n_jobs:
        n_jobs = multiprocessing.cpu_count()
    trees = [tree for _, tree in host.process_trees.items()]
    tasks = []

    for trees_to_check in batch(trees, n_jobs):
        task = pool.apply_async(check_batch, (trees_to_check, new_tree))
        tasks.append(task)
    pool.close()
    pool.join()
    for t in tasks:
        val = t.get()
        if val:
            result += val
    if len(result) > 0:
        return result
    else:
        return None


def find_all_matches_hard(host, new_tree):
    results = []
    for tree_id, tree in host.process_trees.items():
        if tree.matches(new_tree):
            results.append(tree_id)
    if len(results) > 0:
        return results
    else:
        return None


def check_batch(checking_against, searching_for):
    res = []
    for check_this in checking_against:
        if check_this.matches(searching_for):
            res.append(check_this.tree_id)
    if len(res) > 0:
        return res
    else:
        return None


if __name__ == "__main__":
    test_host = read_host('/Users/ajwilson/Sabre_WinEvent_Logs/process_trees/dbc-ib-06.json')
    print(test_host.find_unique_trees(return_dict=True))
    #new_tree = test_host.process_trees[test_host.process_trees]
    #test_host.del_tree(new_tree)
    #start = datetime.datetime.now()
    #print(find_all_matches_hard(test_host, new_tree))
    #print('Hard way in {:,.2f}'.format((datetime.datetime.now() - start).total_seconds()))
