import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import gc
import networkx as nx

def binary_G(G):
    return np.array(G != 0, dtype=np.float)

def compute_G_to_draw(G_pred, commonlabel, no_cycles=False):
    G = nx.convert_matrix.from_numpy_matrix(G_pred, create_using=nx.DiGraph())
    labels = {i: commonlabel[i] for i in G.nodes()}
    if not no_cycles:
        try:
            # cycles = nx.find_cycle(G)
            # for c in cycles:
            #    G.add_edge(c[0], c[1], color='r', weight=2)
            # print('Found Cycles.')
            cycles = nx.simple_cycles(G)
            cycles = list(cycles)
            cycles_tmp = [tuple(c[i:i + 2] for i in range(len(c) - 1)) for c in cycles]
            cycles_new = []
            for ind, c in enumerate(cycles_tmp):
                d = [x for x in c]
                try:
                    d.append([d[-1][1], d[0][0]])
                    cycles_new.append(d)
                except Exception as E:
                    cycles_new.append([[cycles[ind][0], cycles[ind][0]]])  # self-cycle
            for c in cycles_new:
                for e in c:
                    #print(e)
                    G.add_edge(e[0], e[1], color='r', weight=2)
        except Exception as E:
            cycles_new = None
    colors_edges = []
    colors_nodes = ['yellow' for n in G.nodes]
    for u, v in G.edges:
        if u in G[u].keys():
            colors_nodes[u] = 'red'
        if 'color' in G[u][v].keys():
            colors_edges.append(G[u][v]['color'])
        else:
            colors_edges.append('black')
    if not no_cycles and cycles_new is not None:
        num_cycles = len(cycles_new)
    else:
        num_cycles = 0
    num_cycles += len(np.where(colors_nodes == "red")[0])
    pos = nx.circular_layout(G)
    return G, pos, colors_nodes, colors_edges, labels, num_cycles


def plot_digraphs_and_cycles(list_graphs, list_names, commonlabel, alt_size=(10,10), arrowsize=20, font_size=15, node_size=300, no_cycles=False, savefig=None, dpi=None):
    fig, axs = plt.subplots(1, len(list_graphs), figsize=alt_size)
    for ind, g in enumerate(list_graphs):
        G_o, pos_o, colors_nodes_o, colors_edges_o, labels_o, num_cycles_o = compute_G_to_draw(g, commonlabel, no_cycles=no_cycles)
        nx.draw_circular(G_o, ax=axs[ind])
        nx.draw_networkx_nodes(G_o, pos_o, node_color=colors_nodes_o, node_size=node_size, ax=axs[ind])
        nx.draw_networkx_edges(G_o, pos_o, arrowsize=arrowsize, edge_color=colors_edges_o, ax=axs[ind])
        nx.draw_networkx_labels(G_o, pos_o, labels_o, font_size=font_size, ax=axs[ind])
        if not no_cycles:
            axs[ind].set_title('{}\n# of Cycles = {}'.format(list_names[ind], int(num_cycles_o)))
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=dpi)
        print(f"Figure saved at {savefig}")
    else:
        plt.show()
    clean_plt()


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges
    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.
    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def plot_all_individual(l_mat, l_title, suptitle, alt_form=None, alt_size=None, vmin=-1, vmax=1, sharex=False,
                        sharey=False, commonlabel=None, show_text=True, labelfs=8, savefig=None, dpi=None):
    clean_plt()
    plt.set_cmap('RdBu')
    if alt_form:
        y, x = alt_form
    else:
        y = 2
        x = 4
    if alt_size:
        size_x, size_y = alt_size
    else:
        size_x, size_y = (13, 8)
    fig, axs = plt.subplots(y, x, figsize=(size_x, size_y), sharex=sharex, sharey=sharey)

    for ind, a in enumerate(axs.flatten()):
        im = a.imshow(l_mat[ind], vmin=vmin, vmax=vmax)
        a.set_title(l_title[ind])
        if show_text:
            for (j, i), label in np.ndenumerate(l_mat[ind]):
                a.text(i, j, np.round(label, decimals=2), ha='center', va='center')
        if commonlabel:  # assumes same shape matrices and square matrices
            a.set_xticks(np.arange(len(l_mat[ind])))
            a.set_xticklabels(commonlabel, fontsize=labelfs, rotation=90)
            a.set_yticks(np.arange(len(l_mat[ind])))
            a.set_yticklabels(commonlabel, fontsize=labelfs)

    plt.suptitle(suptitle)
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=dpi)
        print(f"Figure saved at {savefig}")
    else:
        plt.show()
    clean_plt()


def clean_plt():
    plt.close('all')
    gc.collect()