# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


##################################### LIBRARIES ###########################################
import warnings
import faiss
from scipy import sparse as sp

from sklearn import metrics
warnings.filterwarnings("ignore")

import numpy as np, time, pickle as pkl, csv
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.preprocessing import normalize

from tqdm import tqdm

import torch
import auxiliaries as aux

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

"""=============================================================================================="""
"""=============================================================================================="""
"""========================================================="""


def evaluate(dataset, LOG, **kwargs):
    """
    Given a dataset name, applies the correct evaluation function.

    Args:
        dataset: str, name of dataset.
        LOG:     aux.LOGGER instance, main logging class.
        **kwargs: Input Argument Dict, depends on dataset.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    if dataset in ['cars196', 'cub200', 'online_products']:
        ret = evaluate_one_dataset(LOG, **kwargs)
    elif dataset in ['in-shop']:
        ret = evaluate_query_and_gallery_dataset(LOG, **kwargs)
    elif dataset in ['vehicle_id']:
        ret = evaluate_multiple_datasets(LOG, **kwargs)
    else:
        raise Exception('No implementation for dataset {} available!')

    return ret


"""========================================================="""


class DistanceMeasure():
    """
    Container class to run and log the change of distance ratios
    between intra-class distances and inter-class distances.
    """

    def __init__(self, checkdata, opt, name='Train', update_epochs=1):
        """
        Args:
            checkdata: PyTorch DataLoader, data to check distance progression.
            opt:       argparse.Namespace, contains all training-specific parameters.
            name:      str, Name of instance. Important for savenames.
            update_epochs:  int, Only compute distance ratios every said epoch.
        Returns:
            Nothing!
        """
        self.update_epochs = update_epochs
        self.pars = opt
        self.save_path = opt.save_path

        self.name = name
        self.csv_file = opt.save_path + '/distance_measures_{}.csv'.format(self.name)
        with open(self.csv_file, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Rel. Intra/Inter Distance'])

        self.checkdata = checkdata

        self.mean_class_dists = []
        self.epochs = []

    def measure(self, model, epoch):
        """
        Compute distance ratios of intra- and interclass distance.

        Args:
            model: PyTorch Network, network that produces the resp. embeddings.
            epoch: Current epoch.
        Returns:
            Nothing!
        """
        if epoch % self.update_epochs: return

        self.epochs.append(epoch)

        torch.cuda.empty_cache()

        _ = model.eval()

        # Compute Embeddings
        with torch.no_grad():
            feature_coll, target_coll = [], []
            data_iter = tqdm(self.checkdata, desc='Estimating Data Distances...')
            for idx, data in enumerate(data_iter):
                input_img, target = data[1], data[0]
                features = model(input_img.to(self.pars.device))
                feature_coll.extend(features.cpu().detach().numpy().tolist())
                target_coll.extend(target.numpy().tolist())

        feature_coll = np.vstack(feature_coll).astype('float32')
        target_coll = np.hstack(target_coll).reshape(-1)
        avail_labels = np.unique(target_coll)

        # Compute indixes of embeddings for each class.
        class_positions = []
        for lab in avail_labels:
            class_positions.append(np.where(target_coll == lab)[0])

        # Compute average intra-class distance and center of mass.
        com_class, dists_class = [], []
        for class_pos in class_positions:
            dists = distance.cdist(feature_coll[class_pos], feature_coll[class_pos], 'cosine')
            dists = np.sum(dists) / (len(dists) ** 2 - len(dists))
            # dists = np.linalg.norm(np.std(feature_coll_aux[class_pos],axis=0).reshape(1,-1)).reshape(-1)
            com = normalize(np.mean(feature_coll[class_pos], axis=0).reshape(1, -1)).reshape(-1)
            dists_class.append(dists)
            com_class.append(com)

        # Compute mean inter-class distances by the class-coms.
        mean_inter_dist = distance.cdist(np.array(com_class), np.array(com_class), 'cosine')
        mean_inter_dist = np.sum(mean_inter_dist) / (
                    len(mean_inter_dist) ** 2 - len(mean_inter_dist))

        # Compute distance ratio
        mean_class_dist = np.mean(np.array(dists_class) / mean_inter_dist)
        self.mean_class_dists.append(mean_class_dist)

        self.update(mean_class_dist)

    def update(self, mean_class_dist):
        """
        Update Loggers.

        Args:
            mean_class_dist: float, Distance Ratio
        Returns:
            Nothing!
        """
        self.update_csv(mean_class_dist)
        self.update_plot()

    def update_csv(self, mean_class_dist):
        """
        Update CSV.

        Args:
            mean_class_dist: float, Distance Ratio
        Returns:
            Nothing!
        """
        with open(self.csv_file, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([mean_class_dist])

    def update_plot(self):
        """
        Update progression plot.

        Args:
            None.
        Returns:
            Nothing!
        """
        plt.style.use('ggplot')
        f, ax = plt.subplots(1)
        ax.set_title('Mean Intra- over Interclassdistances')
        ax.plot(self.epochs, self.mean_class_dists, label='Class')
        f.legend()
        f.set_size_inches(15, 8)
        f.savefig(self.save_path + '/distance_measures_{}.svg'.format(self.name))


class GradientMeasure():
    """
    Container for gradient measure functionalities.
    Measure the gradients coming from the embedding layer to the final conv. layer
    to examine learning signal.
    """

    def __init__(self, opt, name='class-it'):
        """
        Args:
            opt:   argparse.Namespace, contains all training-specific parameters.
            name:  Name of class instance. Important for the savename.
        Returns:
            Nothing!
        """
        self.pars = opt
        self.name = name
        self.saver = {'grad_normal_mean': [], 'grad_normal_std': [], 'grad_abs_mean': [],
                      'grad_abs_std': []}

    def include(self, params):
        """
        Include the gradients for a set of parameters, normally the final embedding layer.

        Args:
            params: PyTorch Network layer after .backward() was called.
        Returns:
            Nothing!
        """
        gradients = [params.weight.grad.detach().cpu().numpy()]

        for grad in gradients:
            ### Shape: 128 x 2048
            self.saver['grad_normal_mean'].append(np.mean(grad, axis=0))
            self.saver['grad_normal_std'].append(np.std(grad, axis=0))
            self.saver['grad_abs_mean'].append(np.mean(np.abs(grad), axis=0))
            self.saver['grad_abs_std'].append(np.std(np.abs(grad), axis=0))

    def dump(self, epoch):
        """
        Append all gradients to a pickle file.

        Args:
            epoch: Current epoch
        Returns:
            Nothing!
        """
        with open(self.pars.save_path + '/grad_dict_{}.pkl'.format(self.name), 'ab') as f:
            pkl.dump([self.saver], f)
        self.saver = {'grad_normal_mean': [], 'grad_normal_std': [], 'grad_abs_mean': [],
                      'grad_abs_std': []}


"""========================================================="""


def evaluate_one_dataset(LOG, dataloader, model, opt, save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        dataloader:  PyTorch Dataloader, Testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    image_paths = np.array(dataloader.dataset.image_list)
    with torch.no_grad():
        # Compute Metrics
        F1, NMI, recall_at_ks, feature_matrix_all = aux.eval_metrics_one_dataset(model, dataloader,
                                                                                 device=opt.device,
                                                                                 k_vals=opt.k_vals,
                                                                                 opt=opt, dim=0, LOG=LOG)
        # print(F1, NMI, recall_at_ks)

        F1_, NMI_, recall_at_ks_, feature_matrix_all_ = aux.eval_metrics_one_dataset(model, dataloader,
                                                                                 device=opt.device,
                                                                                 k_vals=opt.k_vals,
                                                                                 opt=opt, dim=1, LOG=LOG)
        # Make printable summary string.
        result_str = ', '.join(
            '@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch,
                                                                                            NMI, F1,
                                                                                            result_str)

        result_str_ = ', '.join(
            '@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_at_ks_))
        result_str_ = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch,
                                                                                            NMI_, F1_,
                                                                                            result_str_)


        if LOG is not None:
            if save:
                if not len(LOG.progress_saver['val']['Recall @ 1']) or recall_at_ks[0] > np.max(
                        LOG.progress_saver['val']['Recall @ 1']):
                    # Save Checkpoint
                    aux.set_checkpoint(model, opt, LOG.progress_saver,
                                       LOG.prop.save_path + '/checkpoint.pth.tar')
                    aux.recover_closest_one_dataset_myself(feature_matrix_all, image_paths,
                                                    LOG.prop.save_path + '/sample_recoveries.png')
            # Update logs.
            LOG.log('val', LOG.metrics_to_log['val'],
                    [epoch, np.round(time.time() - start), NMI, F1] + recall_at_ks)

            LOG.log('val2', LOG.metrics_to_log['val2'],
                    [epoch, np.round(time.time() - start), NMI_, F1_] + recall_at_ks_)

    print(result_str)
    print(result_str_)
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None


"""========================================================="""


def evaluate_query_and_gallery_dataset(LOG, query_dataloader, gallery_dataloader, model, opt,
                                       save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for In-Shop Clothes.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        query_dataloader:    PyTorch Dataloader, Query-testdata to be evaluated.
        gallery_dataloader:  PyTorch Dataloader, Gallery-testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    query_image_paths = np.array([x[0] for x in query_dataloader.dataset.image_list])
    gallery_image_paths = np.array([x[0] for x in gallery_dataloader.dataset.image_list])

    with torch.no_grad():
        # Compute Metrics.
        F1, NMI, recall_at_ks, query_feature_matrix_all, gallery_feature_matrix_all = aux.eval_metrics_query_and_gallery_dataset(
            model, query_dataloader, gallery_dataloader, device=opt.device, k_vals=opt.k_vals,
            opt=opt)
        # Generate printable summary string.
        result_str = ', '.join(
            '@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch,
                                                                                            NMI, F1,
                                                                                            result_str)

        if LOG is not None:
            if save:
                if not len(LOG.progress_saver['val']['Recall @ 1']) or recall_at_ks[0] > np.max(
                        LOG.progress_saver['val']['Recall @ 1']):
                    # Save Checkpoint
                    aux.set_checkpoint(model, opt, LOG.progress_saver,
                                       LOG.prop.save_path + '/checkpoint.pth.tar')
                    aux.recover_closest_inshop(query_feature_matrix_all, gallery_feature_matrix_all,
                                               query_image_paths, gallery_image_paths,
                                               LOG.prop.save_path + '/sample_recoveries.png')
            # Update logs.
            LOG.log('val', LOG.metrics_to_log['val'],
                    [epoch, np.round(time.time() - start), NMI, F1] + recall_at_ks)

    print(result_str)
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None


"""========================================================="""


def evaluate_multiple_datasets(LOG, dataloaders, model, opt, save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for Multi-test datasets s.a. PKU Vehicle ID.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        dataloaders: List of PyTorch Dataloaders, test-dataloaders to evaluate.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()

    csv_data = [epoch]

    with torch.no_grad():
        for i, dataloader in enumerate(dataloaders):
            print('Working on Set {}/{}'.format(i + 1, len(dataloaders)))
            image_paths = np.array(dataloader.dataset.image_list)
            # Compute Metrics for specific testset.
            F1, NMI, recall_at_ks, feature_matrix_all = aux.eval_metrics_one_dataset(model,
                                                                                     dataloader,
                                                                                     device=opt.device,
                                                                                     k_vals=opt.k_vals,
                                                                                     opt=opt)
            # Generate printable summary string.
            result_str = ', '.join(
                '@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_at_ks))
            result_str = 'SET {0}: Epoch (Test) {1}: NMI [{2:.4f}] | F1 {3:.4f}| Recall [{4}]'.format(
                i + 1, epoch, NMI, F1, result_str)

            if LOG is not None:
                if save:
                    if not len(LOG.progress_saver['val']['Set {} Recall @ 1'.format(i)]) or \
                            recall_at_ks[0] > np.max(
                            LOG.progress_saver['val']['Set {} Recall @ 1'.format(i)]):
                        # Save Checkpoint for specific test set.
                        aux.set_checkpoint(model, opt, LOG.progress_saver,
                                           LOG.prop.save_path + '/checkpoint_set{}.pth.tar'.format(
                                               i + 1))
                        aux.recover_closest_one_dataset(feature_matrix_all, image_paths,
                                                        LOG.prop.save_path + '/sample1_recoveries_set{}.png'.format(
                                                            i + 1))
                        aux.recover_closest_one_dataset_myself(feature_matrix_all, image_paths,
                                                         LOG.prop.save_path + '/my_sample_recoveries_set{}.png'.format(
                                                             i + 1))

                csv_data += [NMI, F1] + recall_at_ks
            print(result_str)

    csv_data.insert(0, np.round(time.time() - start))
    # Update logs.
    LOG.log('val', LOG.metrics_to_log['val'], csv_data)

    if give_return:
        return csv_data[2:]
    else:
        None




def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency





def evaluate_classification(dataloader, model, opt, criterion_CE):

    # image_paths = np.array(dataloader.dataset.image_list)
    torch.cuda.empty_cache()

    _ = model.eval()
    # print(dataloader.dataset.__len__())
    COUNT = 0
    TRUE_LABEL=0
    with torch.no_grad():
        ### For all test images, extract features
        final_iter = tqdm(dataloader, desc='Computing Evaluation Metrics for classification ...')

        for idx, inp in enumerate(final_iter):
            # print(idx)

            input_img, target = inp[-1], inp[0]
            COUNT += len(input_img)
            _, out, _ = model(input_img.to(opt.device))
            # print(out.size())
            loss_cer = criterion_CE(out, target)
            # print(out.size())    torch.Size([64, 226])
            pred = torch.argmax(out, dim=1)

            # print(pred.size()) #torch.Size([64])
            # print(pred.detach().cpu().numpy())
            # print(target.detach().cpu().numpy())
            TRUE_LABEL += len(np.where(pred.detach().cpu().numpy() == target.numpy())[0])

    print((TRUE_LABEL * 100) / COUNT)
    torch.cuda.empty_cache()
    return loss_cer.item()


def find_mus(train_data, model, opt, LOG= None):

    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(train_data.dataset.avail_classes)
    print(n_classes)

    with torch.no_grad():
        ### For all test images, extract features
        target_labels, feature_coll = [], []
        final_iter = tqdm(train_data, desc='finding muuu for each class ...')
        # image_paths = [x[0] for x in train_data.dataset.image_list]

        for idx, inp in enumerate(final_iter):
            input_img, target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out, _, _ = model(input_img.to(opt.device))

            feature_coll.extend(out.cpu().detach().numpy().tolist())

        # print(len(target_labels), len(feature_coll))  #4032
        # print(len(feature_coll[0]))    #256

        if LOG:
            np.save(LOG.prop.save_path + "/train_embedding.npy", np.array(feature_coll))
            np.save(LOG.prop.save_path + "/train_label.npy", np.array(target_labels))
            print("embedding and label saved ...")

        target_labels = np.hstack(target_labels).reshape(-1, 1)
        feature_coll = np.vstack(feature_coll).astype('float32')

        torch.cuda.empty_cache()
        # ## Set Faiss CPU Cluster index
        cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        kmeans = faiss.Clustering(feature_coll.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        # ## Train Kmeans
        kmeans.train(feature_coll, cpu_cluster_index)

        # print(len(faiss.vector_float_to_array(kmeans.centroids)))
        # print(faiss.vector_float_to_array(kmeans.centroids))
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes,
                                                                                   feature_coll.shape[
                                                                                       -1])
        # print(computed_centroids)
        # print(computed_centroids.shape)  #(49, 256)

        # ## Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        # print(faiss_search_index)
        faiss_search_index.add(computed_centroids)
        # print(faiss_search_index)
        _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)
        # print(model_generated_cluster_labels)  #list[list[int]]   [[2],[1],[7], ...]
        # print(model_generated_cluster_labels.shape)   #(4032, 1)

        model_generated_cluster_labels = model_generated_cluster_labels.reshape(-1)
        # print(model_generated_cluster_labels.shape)
        cluster_labels = []
        aa=contingency_matrix(target_labels, model_generated_cluster_labels, eps=None, sparse=True)
        # find_class(model_generated_cluster_labels, target_labels)
        print(aa)
        hhhh
        for i in range(n_classes):
            indices = np.where(model_generated_cluster_labels == i)[0]
            print(indices)
            print(indices.shape)
            cluster_y = target_labels[indices]
            print(cluster_y)
            counts = np.bincount(cluster_y.reshape(-1))
            print(counts)
            print(counts.shape)
            ddd
            cluster_label = np.argsort(counts, axis=1)[:, -5:]
            # cluster_label = np.argmax(counts)
            print(cluster_label)
            cluster_labels.append(cluster_label)

        # Print the cluster labels
        print("Cluster Labels:", sorted(cluster_labels))
        dddd

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def find_class(cluster_labels, real_labels):

    # Compute the Adjusted Rand Index between the cluster labels and real labels

    cluster_labels = cluster_labels.reshape(-1)
    real_labels = real_labels.reshape(-1)
    ari_score = metrics.adjusted_rand_score(real_labels, cluster_labels)

    # Create a matrix to represent the mapping probabilities between clusters and classes
    num_clusters = len(set(cluster_labels))
    num_classes = len(set(real_labels))
    mapping_matrix = [[0.0 for j in range(num_classes)] for i in range(num_clusters)]

    # Fill in the matrix based on the similarity between the clusters and classes
    for i in range(num_clusters):
        for j in range(num_classes):
            count = 0
            for k in range(len(real_labels)):
                if cluster_labels[k] == i and real_labels[k] == j:
                    count += 1
            prob = float(count) / float(len(real_labels))
            mapping_matrix[i][j] = prob * ari_score

    # Assign each cluster to a class based on the mapping probabilities
    cluster_to_class = {}
    for i in range(num_clusters):
        max_prob = 0.0
        max_class = -1
        for j in range(num_classes):
            if mapping_matrix[i][j] > max_prob:
                max_prob = mapping_matrix[i][j]
                max_class = j
        cluster_to_class[i] = max_class

    # Print the mapping between clusters and classes
    print(cluster_to_class)
    # for i in range(num_clusters):
    #     print("Cluster %d -> Class %d" % (i, cluster_to_class[i]))
    for cluster, cls in cluster_to_class.items():
        print(f"cluster {cluster} : clas {cls}")
    # print(cluster_to_class.keys())
    key=[i for i in cluster_to_class.keys()]
    print(sorted(key))
    val=[ i for _,i in cluster_to_class.items()]
    print(sorted(val))