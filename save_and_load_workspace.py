import pickle


def save_workspace(workspace, filename):
    with open(filename, 'wb') as file:
        pickle.dump(workspace, file)


def load_workspace(filename):
    with open(filename, 'rb') as file:
        workspace = pickle.load(file)
    return workspace
