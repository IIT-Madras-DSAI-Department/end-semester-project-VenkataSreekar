import numpy as np

class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        # Initialize all these as None
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit PCA on the dataset X.
        """
        # 0. Convert to array
        X = np.array(X, dtype=float)
        
        # 1. Center the data
        # compute mean 
        self.mean = np.mean(X, axis=0)
        # center around mean
        X_centered = X - self.mean

        # 2. Covariance matrix
        # compute variance - each feature in columns - rowvar is False
        # rowvar False means features are columns, True means features are rows
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        # compute the eigen values and vectors of covariance martix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvectors by descending eigenvalues
        # sort in descending order of eigen values
        sorted_idx = np.argsort(eigenvalues)[::-1]

        # take top n components
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]

        # get the components 
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):
        """
        Project the data X onto the principal components.
        """
        if self.mean is None or self.components is None:
            # eigen values or vectors do not exist if model is not fitted yet
            raise ValueError("The PCA model has not been fitted yet.")

        # center the data around zero
        X_centered = X - self.mean
        
        # return the dot product with eigen vectors, to retrieve components
        return np.dot(X_centered, self.components)

    def reconstruct(self, X):
        """
        Reconstruct the original data from the reduced representation.
        """
        # predict projections
        Z = self.predict(X)  # Projected data
        
        # reconstruct X based on projects onto components, and add mean value 
        return np.dot(Z, self.components.T) + self.mean

    def detect_anomalies(self, X, threshold=None, return_errors=False):
        """
        Detect anomalies based on reconstruction error.

        Parameters:
        - X: Input data
        - threshold: Optional. If not provided, uses 95th percentile of reconstruction errors
        - return_errors: If True, also returns reconstruction errors

        Returns:
        - is_anomaly: Boolean mask of anomalies
        - errors (optional): Reconstruction errors for each point
        """
        # reconstrut X 
        X_reconstructed = self.reconstruct(X)

        # compute reconstruction error
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        # if no threshold, take 95% value
        if threshold is None:
            threshold = np.percentile(errors, 95)

        # flag observations, whose reconstruction error is higher than this threshold
        flag = errors > threshold
        
        is_anomaly = flag * 1
        
        return is_anomaly, errors
    
    def transform(self, X):
        """
        'transform' is the standard name for the projection operation.
        We just call your predict method, which does the same thing.
        """
        return self.predict(X)

    def fit_transform(self, X):
        """
        Helper method to fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)

class KNN:
    def __init__(self, k=5, batch_size=256):
        self.k = k
        self.batch_size = batch_size
        self.Xtrain = None
        self.ytrain = None
        self.train_norm = None

    def fit(self, X, y):
        """
        Store training data and precompute ||X_train||^2 for fast distance calc.
        """
        self.Xtrain = X.astype(np.float32)
        self.ytrain = y.astype(int)
        self.train_norm = np.sum(self.Xtrain**2, axis=1)  # shape (M,)

    def _compute_distances_batch(self, Xbatch):
        """
        Compute all pairwise distances between Xbatch and Xtrain using:
        ||x - t||^2 = ||x||^2 + ||t||^2 - 2 * x * t^T
        """
        X_norm = np.sum(Xbatch**2, axis=1).reshape(-1, 1)   # shape (B,1)
        cross_term = Xbatch @ self.Xtrain.T                 # shape (B,M)
        dists = X_norm + self.train_norm - 2 * cross_term   # shape (B,M)
        return dists

    def predict(self, X):
        X = X.astype(np.float32)
        n_samples = X.shape[0]
        preds = []

        # process test data in batches (saves memory, much faster)
        for i in range(0, n_samples, self.batch_size):
            Xbatch = X[i:i+self.batch_size]
            dists = self._compute_distances_batch(Xbatch)

            # For each row in batch:
            idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
            neighbor_labels = self.ytrain[idx]

            # majority vote per test sample
            batch_preds = [np.bincount(row, minlength=10).argmax()
                           for row in neighbor_labels]

            preds.extend(batch_preds)

        return np.array(preds, dtype=int)
    

class WeightedKNN:
    def __init__(self, k=5, eps=1e-8, batch_size=256):
        self.k = k
        self.eps = eps
        self.batch_size = batch_size
        self.Xtrain = None
        self.ytrain = None
        self.train_norm = None

    def fit(self, X, y):
        """
        Store training data and precompute ||Xtrain||^2.
        """
        self.Xtrain = np.asarray(X, dtype=np.float32)
        self.ytrain = np.asarray(y, dtype=int)   # MUST be numpy array
        self.train_norm = np.sum(self.Xtrain**2, axis=1)

    def _compute_distances_batch(self, Xbatch):
        """
        Compute squared distances using:
        ||x - t||^2 = ||x||^2 + ||t||^2 - 2 x t^T
        """
        X_norm = np.sum(Xbatch**2, axis=1).reshape(-1, 1)
        cross_term = Xbatch @ self.Xtrain.T
        dists = X_norm + self.train_norm - 2 * cross_term
        return dists

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        preds = []

        for i in range(0, n_samples, self.batch_size):
            Xbatch = X[i:i+self.batch_size]

            # Compute distance matrix for batch
            dists = self._compute_distances_batch(Xbatch)

            # Get top-k smallest distances
            idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]

            nearest_dist = dists[np.arange(len(idx))[:, None], idx]
            nearest_labels = self.ytrain[idx]   # shape (batch_size, k)

            # Compute weights: 1 / (distance + eps)
            weights = 1.0 / (nearest_dist + self.eps)

            # Weighted vote
            batch_preds = []
            for labels_row, weights_row in zip(nearest_labels, weights):
                # sum weights per class (label)
                scores = np.bincount(labels_row, weights_row, minlength=10)
                batch_preds.append(np.argmax(scores))

            preds.extend(batch_preds)

        return np.array(preds, dtype=int)


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None    #weights is a 2D vector
        self.b = None    # bias is also a 2D vector 

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def _cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = np.max(y) + 1  # assuming labels are 0-indexed

        # Initialize weights and bias
        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        #print ('shape of weights is', np.shape(self.W))
        #print ('shape of bias is', np.shape(self.b))

        # One-hot encode labels
        Y_onehot = self._one_hot(y, num_classes)
        #print ('shape of label vector is', np.shape(Y_onehot))

        for epoch in range(self.epochs):
            # Forward pass
            logits = np.dot(X, self.W) + self.b
            probs = self._softmax(logits)
            #print ('shape of logits is', np.shape(logits))
            #print ('shape of probs is', np.shape(probs))

            # Loss (for monitoring)
            loss = self._cross_entropy_loss(Y_onehot, probs)
            #print ('shape of loss vector is', np.shape(loss))

            # Backward pass
            grad_logits = (1./ num_samples) * (Y_onehot - probs) 
            grad_W = -np.dot(X.T, grad_logits)
            grad_b = -np.sum(grad_logits, axis=0, keepdims=True)

            # Update weights
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)



def _rng(seed):
    return np.random.RandomState(seed) if seed is not None else np.random

class DecisionTree:
    """
    CART classifier (Gini) used by RandomForestScratch.
    Supports:
      - max_depth
            if feat is None:
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id
      - min_samples_split
      - min_samples_leaf
      - max_features: 'sqrt' | 'log2' | int | float in (0,1]

    OPTIMIZED:
      - `fit` now accepts indices and does not copy X.
      - `predict_proba` is vectorized.
    """
    __slots__ = (
        "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "n_classes_", "n_features_", "rng",
        "tree_", "X_", "y_"
    )

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.rng = _rng(random_state)

        self.n_classes_ = None
        self.n_features_ = None
        self.X_ = None  # Stores pointer to the original X
        self.y_ = None  # Stores pointer to the original y

        # --- OPTIMIZED: Tree is stored as parallel arrays ---
        self.tree_ = {} # This will hold the NumPy arrays

    # ------- helpers -------
    def _majority_proba(self, y_indices, n_classes):
        if y_indices.size == 0:
            return np.ones(n_classes) / n_classes
        counts = np.bincount(self.y_[y_indices], minlength=n_classes).astype(float)
        return counts / counts.sum()

    def _gini_from_counts(self, counts):
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        return 1.0 - np.sum(p * p)

    def _choose_feature_subset(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features string must be 'sqrt' or 'log2'")
        elif isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features float must be in (0, 1].")
            k = max(1, int(self.max_features * n_features))
        else:
            k = int(self.max_features)
            if k <= 0 or k > n_features:
                k = n_features
        return self.rng.choice(n_features, size=k, replace=False)

    # -------- exact split search (sort + scan all change points) --------
    def _best_split_for_feature(self, X_col_subset, y_subset, n_classes, min_leaf):
        """
        This function is already highly optimized (vectorized Gini calc).
        It takes 1D arrays (copies) which is fine and fast.
        """
        n = y_subset.shape[0]
        if n < 2 * min_leaf:
            return np.inf, None, None

        order = np.argsort(X_col_subset, kind="mergesort")
        x_sorted = X_col_subset[order]
        y_sorted = y_subset[order]

        if x_sorted[0] == x_sorted[-1]:
            return np.inf, None, None

        left_counts = np.zeros((n, n_classes), dtype=np.int32)
        for c in range(n_classes):
            left_counts[:, c] = np.cumsum((y_sorted == c).astype(np.int32))
        total_counts = left_counts[-1].copy()

        diffs = np.diff(x_sorted)
        valid_pos = np.where(diffs > 0)[0]
        if valid_pos.size == 0:
            return np.inf, None, None

        left_sizes = valid_pos + 1
        right_sizes = n - left_sizes
        ok = (left_sizes >= min_leaf) & (right_sizes >= min_leaf)
        valid_pos = valid_pos[ok]
        if valid_pos.size == 0:
            return np.inf, None, None

        lc = left_counts[valid_pos]
        rc = total_counts[None, :] - lc
        lsz = left_sizes[ok][:, None].astype(float)
        rsz = right_sizes[ok][:, None].astype(float)
        n_tot = (lsz + rsz)

        l_gini = 1.0 - np.sum((lc / lsz) ** 2, axis=1)
        r_gini = 1.0 - np.sum((rc / rsz) ** 2, axis=1)
        weighted = (lsz[:, 0] / n_tot[:, 0]) * l_gini + (rsz[:, 0] / n_tot[:, 0]) * r_gini

        best_idx = np.argmin(weighted)
        best_pos = valid_pos[best_idx]
        thr = 0.5 * (x_sorted[best_pos] + x_sorted[best_pos + 1])

        left_mask = X_col_subset <= thr
        if (np.sum(left_mask) < min_leaf) or (n - np.sum(left_mask) < min_leaf):
            return np.inf, None, None

        return weighted[best_idx], float(thr), left_mask

    def _best_split(self, node_X_idx):
        """
        OPTIMIZED: Operates on indices `node_X_idx`.
        It only copies 1D arrays for y and a single feature column.
        """
        n_samples = node_X_idx.size
        n_classes = self.n_classes_

        # Fast 1D copy
        y_node = self.y_[node_X_idx]

        feat_idx = self._choose_feature_subset(self.n_features_)

        parent_counts = np.bincount(y_node, minlength=n_classes)
        parent_gini = self._gini_from_counts(parent_counts)
        best = (np.inf, None, None, None) # (impurity, feat, thr, left_mask)

        if parent_gini == 0.0:
            return best

        for j in feat_idx:
            # Fast 1D copy
            X_col_node = self.X_[node_X_idx, j]

            impurity, thr, left_mask = self._best_split_for_feature(
                X_col_node, y_node, n_classes, self.min_samples_leaf
            )
            if impurity < best[0]:
                best = (impurity, j, thr, left_mask)

        return best

    # -------- training & inference --------
    def fit(self, X, y, sample_indices=None, n_classes=None):
        """
        OPTIMIZED:
        `X` and `y` are the *full* datasets.
        `sample_indices` are the bootstrap indices.
        This avoids copying X and y for every tree.
        """
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y, dtype=np.int32)
        self.n_features_ = self.X_.shape[1]

        if n_classes is None:
            n_classes = int(np.max(self.y_)) + 1
        self.n_classes_ = n_classes

        if sample_indices is None:
            root_indices = np.arange(self.X_.shape[0])
        else:
            root_indices = sample_indices
        
        # This list-of-dicts is temporary and will be converted
        # for fast prediction.
        nodes_list = []

        def _build(node_X_idx, depth):
            y_node_size = node_X_idx.size # Use size for speed
            
            # Use 1D copy only for stats
            y_node_for_stats = self.y_[node_X_idx]
            proba = self._majority_proba(node_X_idx, self.n_classes_)

            if (self.max_depth is not None and depth >= self.max_depth) \
                or (y_node_size < self.min_samples_split) \
                or (np.unique(y_node_for_stats).size == 1):
                
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id

            impurity, feat, thr, left_mask = self._best_split(node_X_idx)

            if feat is None:
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id

            left_idx_global = node_X_idx[left_mask]
            right_idx_global = node_X_idx[~left_mask]

            if left_idx_global.size < self.min_samples_leaf or right_idx_global.size < self.min_samples_leaf:
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id

            left_child = _build(left_idx_global, depth + 1)
            right_child = _build(right_idx_global, depth + 1)

            node_id = len(nodes_list)
            nodes_list.append({
                "is_leaf": False,
                "proba": proba,
                "feat": feat,
                "thr": float(thr),
                "left": left_child,
                "right": right_child
            })
            return node_id

        root_id = _build(root_indices, depth=0)
        
        # --- OPTIMIZATION: Convert list-of-dicts to NumPy arrays ---
        n_nodes = len(nodes_list)
        self.tree_['is_leaf'] = np.zeros(n_nodes, dtype=bool)
        self.tree_['proba'] = np.zeros((n_nodes, self.n_classes_), dtype=float)
        self.tree_['feat'] = np.full(n_nodes, -1, dtype=np.int32)
        self.tree_['thr'] = np.zeros(n_nodes, dtype=float)
        self.tree_['left'] = np.full(n_nodes, -1, dtype=np.int32)
        self.tree_['right'] = np.full(n_nodes, -1, dtype=np.int32)
        self.tree_['root'] = root_id

        for i, node_d in enumerate(nodes_list):
            self.tree_['is_leaf'][i] = node_d['is_leaf']
            self.tree_['proba'][i, :] = node_d['proba']
            self.tree_['feat'][i] = node_d['feat']
            self.tree_['thr'][i] = node_d['thr']
            self.tree_['left'][i] = node_d['left']
            self.tree_['right'][i] = node_d['right']

        # We no longer need these pointers
        del self.X_
        del self.y_
        return self

    def predict_proba(self, X):
        """
        OPTIMIZED: Vectorized tree traversal.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Get pointers to the tree arrays
        is_leaf = self.tree_['is_leaf']
        feat = self.tree_['feat']
        thr = self.tree_['thr']
        left = self.tree_['left']
        right = self.tree_['right']
        proba = self.tree_['proba']
        root = self.tree_['root']

        # `node_ids` tracks the current node for *each* sample
        node_ids = np.full(n_samples, root, dtype=np.int32)
        # `probas` will store the result for each sample
        probas = np.zeros((n_samples, self.n_classes_))
        
        # `active` mask tracks which samples are still traversing
        active = np.ones(n_samples, dtype=bool)

        # --- FIX: Re-written while loop to prevent IndexError ---
        while np.any(active):
            # Get all samples that are still active
            active_idx = np.where(active)[0]
            
            # Get their current node IDs
            curr_nodes = node_ids[active_idx]
            
            # Check which of these are leaf nodes
            # `leaves_mask` is a boolean array of shape (len(active_idx),)
            leaves_mask = is_leaf[curr_nodes]
            
            # --- Process the leaves ---
            if np.any(leaves_mask):
                # Get the *original* indices of the samples that are at a leaf
                leaf_indices = active_idx[leaves_mask]
                
                # Assign their probabilities
                probas[leaf_indices] = proba[node_ids[leaf_indices]]
                
                # Deactivate them from future loops
                active[leaf_indices] = False

            # --- Process the non-leaves ---
            # Check if any samples are *still* active
            if not np.any(active):
                break # All done

            # Get the *original* indices of non-leaf samples
            non_leaf_mask = ~leaves_mask
            non_leaf_indices = active_idx[non_leaf_mask]
            
            # Get the node IDs for *only* the non-leaf samples
            non_leaf_node_ids = node_ids[non_leaf_indices]

            # Get features and thresholds for these nodes
            node_feats = feat[non_leaf_node_ids]
            node_thrs = thr[non_leaf_node_ids]
            
            # Get the corresponding data from X and make the decision
            X_data = X[non_leaf_indices, node_feats]
            go_left_mask = X_data <= node_thrs

            # Get the global indices for left/right branches
            left_global_indices = non_leaf_indices[go_left_mask]
            right_global_indices = non_leaf_indices[~go_left_mask]
            
            # Get the children node IDs
            left_child_nodes = left[non_leaf_node_ids[go_left_mask]]
            right_child_nodes = right[non_leaf_node_ids[~go_left_mask]]

            # Update node_ids for the next iteration
            node_ids[left_global_indices] = left_child_nodes
            node_ids[right_global_indices] = right_child_nodes
            
        return probas

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class RandomForrest:
    """
    Random Forest built on DecisionTreeScratch.
    Supports:
      - n_estimators, max_depth, max_features
      - min_samples_leaf, min_samples_split
      - bootstrap, max_samples, random_state
    
    OPTIMIZED:
      - `fit` now passes indices, avoiding (N, D) copies.
    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 max_features="sqrt",
                 min_samples_leaf=1,
                 min_samples_split=2,
                 bootstrap=True,
                 max_samples=None,
                 random_state=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.random_state = random_state

        self.n_classes_ = None
        self.trees_ = []
        self.rng = _rng(random_state)

    def _draw_sample_indices(self, n_samples):
        if self.max_samples is None:
            m = n_samples
        elif isinstance(self.max_samples, float):
            if not (0.0 < self.max_samples <= 1.0):
                raise ValueError("max_samples float must be in (0, 1].")
            m = max(1, int(self.max_samples * n_samples))
        else:
            m = int(self.max_samples)
            if m <= 0 or m > n_samples:
                m = n_samples

        if self.bootstrap:
            return self.rng.randint(0, n_samples, size=m)  # with replacement
        else:
            return self.rng.choice(n_samples, size=m, replace=False)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int32)
        self.n_classes_ = int(np.max(y)) + 1
        self.trees_ = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            print(f"Fitting tree {_ + 1}/{self.n_estimators}")
            
            # --- OPTIMIZATION: Get indices, don't copy data ---
            idx = self._draw_sample_indices(n_samples)
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.rng.randint(0, 2**31 - 1)
            )
            
            # --- OPTIMIZATION: Pass full X, y and indices ---
            tree.fit(X, y, sample_indices=idx, n_classes=self.n_classes_)
            
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if not self.trees_:
            raise RuntimeError("Model not fitted.")
        
        proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        
        # This loop is now the main "slow" part, but it's
        # calling the fast, vectorized tree.predict_proba.
        for tree in self.trees_:
            proba_sum += tree.predict_proba(X)
            
        return proba_sum / len(self.trees_)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    

import numpy as np

class treeNode:
    def __init__(self, threshold=None, feature_index=None, value=None):
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value
        self.left = None
        self.right = None

    def is_leaf_Node(self):
        return self.value is not None


class XGBoostClassifier:
    def __init__(self, n_estimators=500, learning_rate=0.5, max_depth=6,
                 lamda=3.0, subsample_features=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamda = lamda
        self.subsample_features = subsample_features
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _g(self, y_true, y_pred):
        return self._sigmoid(y_pred) - y_true

    def _h(self, y_true, y_pred):
        sig = self._sigmoid(y_pred)
        return sig * (1 - sig)

    def _exact_greedysplit_vectorized(self, X_col, y_true, y_pred):
        """Sparse-aware, vectorized split finder."""
        g = self._g(y_true, y_pred)
        h = self._h(y_true, y_pred)

        # skip all-zero columns
        nonzero_mask = X_col != 0
        X_nz = X_col[nonzero_mask]
        g_nz = g[nonzero_mask]
        h_nz = h[nonzero_mask]

        if X_nz.size < 2:
            return -np.inf, None

        # Total sums
        G_total, H_total = np.sum(g), np.sum(h)

        # Zero-entry contribution
        G_zero = np.sum(g[~nonzero_mask])
        H_zero = np.sum(h[~nonzero_mask])

        # Sort once
        sorted_idx = np.argsort(X_nz)
        X_sorted = X_nz[sorted_idx]
        g_sorted = g_nz[sorted_idx]
        h_sorted = h_nz[sorted_idx]

        # Vectorized prefix sums (no explicit loop)
        G_L = G_zero + np.cumsum(g_sorted)
        H_L = H_zero + np.cumsum(h_sorted)
        G_R = G_total - G_L
        H_R = H_total - H_L

        # Gain for all splits at once
        gain = (G_L**2) / (H_L + self.lamda + 1e-6) + \
               (G_R**2) / (H_R + self.lamda + 1e-6) - \
               (G_total**2) / (H_total + self.lamda + 1e-6)

        best_idx = np.argmax(gain)
        best_gain = gain[best_idx]
        best_threshold = X_sorted[best_idx]

        return best_gain, best_threshold

    def _build_tree(self, X, y_true, y_pred, depth):
        n_samples, n_features = X.shape
        if (n_samples < 3) or (depth >= self.max_depth):
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)

        # Random feature subsampling
        feature_indices = np.random.choice(
            n_features,
            int(max(1, self.subsample_features * n_features)),
            replace=False
        )

        best_gain, best_threshold, best_feature = -np.inf, None, None

        for feature_index in feature_indices:
            gain, threshold = self._exact_greedysplit_vectorized(X[:, feature_index], y_true, y_pred)
            if gain > best_gain:
                best_gain, best_threshold, best_feature = gain, threshold, feature_index

        if best_gain < 1e-6:
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_mask], y_true[left_mask], y_pred[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y_true[right_mask], y_pred[right_mask], depth + 1)

        node = treeNode(threshold=best_threshold, feature_index=best_feature)
        node.left = left_subtree
        node.right = right_subtree
        return node

    def _predict_tree(self, X, tree):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = tree
            while not node.is_leaf_Node():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.value
        return y_pred

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_mean = np.mean(y)
        y_pred = np.full(y.shape, np.log(y_mean / (1 - y_mean + 1e-6)))

        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y, y_pred, 0)
            self.trees.append(tree)
            update = self._predict_tree(X, tree)
            y_pred += self.learning_rate * update

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(X, tree)
        y_pred = self._sigmoid(y_pred)
        return y_pred


class OneVsAllXGBoost:
    """
    One-vs-All multi-class classifier using XGBoostClassifier as base estimator
    """
    def __init__(self, n_estimators=50, learning_rate=0.3, max_depth=6,
                 lamda=3.0, subsample_features=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamda = lamda
        self.subsample_features = subsample_features
        self.classifiers = []
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Train one binary classifier for each class
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels (0 to n_classes-1)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.classifiers = []
        
        print(f"Training One-vs-All classifiers for {len(self.classes_)} classes...")
        
        for i, cls in enumerate(self.classes_):
            print(f"Training classifier for class {cls} ({i+1}/{len(self.classes_)})")
            
            # Create binary labels: 1 for current class, 0 for all others
            y_binary = (y == cls).astype(int)
            
            # Initialize and train binary classifier
            classifier = XGBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                lamda=self.lamda,
                subsample_features=self.subsample_features
            )
            
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)
        
        print("One-vs-All training completed!")
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classifiers)
        
        # Get probabilities from each binary classifier
        proba_matrix = np.zeros((n_samples, n_classes))
        
        for i, classifier in enumerate(self.classifiers):
            proba_matrix[:, i] = classifier.predict(X)
        
        # Normalize probabilities to sum to 1 for each sample
        # Using softmax-like normalization
        proba_sum = np.sum(proba_matrix, axis=1, keepdims=True)
        proba_matrix = proba_matrix / (proba_sum + 1e-10)
        
        return proba_matrix
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
