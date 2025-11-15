from collections import defaultdict
import numpy as np

class WeightedEnsemble:
    """
    An ensemble that internally handles PCA based on a flag list.
    
    Args:
        model_weights_dict (dict): {'name': (model_obj, weight)}
        pca_flags (list): A list of 0s and 1s, same length as model_dict.
                          1 = Apply PCA, 0 = Do not apply PCA.
        pca_class (class): The *class* (not an object) to use for PCA, 
                           e.g., PCAModel (from Part 1).
        pca_components (int): The number of components to use for PCA.
    """
    def __init__(self, model_weights_dict, pca_flags, pca_class, pca_components=49):
        if len(model_weights_dict) != len(pca_flags):
            raise ValueError("The model dictionary and pca_flags list must be the same length.")
            
        self.model_weights_dict = model_weights_dict
        self.pca_flags = pca_flags
        self.pca_class = pca_class # Your PCAModel class
        self.pca_components = pca_components
        
        self.fitted_models = {}
        self.fitted_pcas = {} # Stores a fitted PCA for each model that needs one

    def _manual_weighted_mode(self, predictions_stack, weights_array):
        """Calculates the weighted mode from a stack of predictions."""
        n_models, n_samples = predictions_stack.shape
        final_preds = []
        
        for i in range(n_samples): # For each sample
            weight_counts = defaultdict(float)
            for j in range(n_models): # For each model
                pred = predictions_stack[j, i]
                weight = weights_array[j]
                weight_counts[pred] += weight
            
            best_value = max(weight_counts, key=weight_counts.get)
            final_preds.append(best_value)
        return np.array(final_preds)

    def fit(self, X_train, y_train):
        """
        Fits all models. If a model's flag is 1, it will
        fit a new PCA and then fit the model on the transformed data.
        """
        print("--- Fitting Ensemble Models ---")
        self.fitted_models = {}
        self.fitted_pcas = {}
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        model_names = list(self.model_weights_dict.keys())
        
        for i in range(len(model_names)):
            name = model_names[i]
            model, weight = self.model_weights_dict[name]
            flag = self.pca_flags[i]
            
            if flag == 1:
                # --- This model NEEDS PCA ---
                pca = self.pca_class(n_components=self.pca_components)
                
                # This is why we added .fit_transform()
                X_train_transformed = pca.fit_transform(X_train)
                
                self.fitted_pcas[name] = pca
                
                print(f"Fitting {name} (weight: {weight}) on PCA data...")
                model.fit(X_train_transformed, y_train)
                
            else:
                # --- This model does NOT need PCA ---
                print(f"Fitting {name} (weight: {weight}) on raw data...")
                model.fit(X_train, y_train)
                
            self.fitted_models[name] = model
            
        print("--- All models fitted. ---")

    def predict(self, X_val):
        """
        Predicts on new data (X_val). If a model was trained
        on PCA, this will transform X_val before predicting.
        """
        if not self.fitted_models:
            raise RuntimeError("Ensemble must be fitted before predicting. Call .fit() first.")

        print("\n--- Generating predictions from all models ---")
        all_predictions = []
        all_weights = []
        X_val = np.array(X_val)

        for name, (original_model, weight) in self.model_weights_dict.items():
            fitted_model = self.fitted_models[name]
            
            if name in self.fitted_pcas:
                # --- This model USED PCA ---
                pca = self.fitted_pcas[name]
                
                # This is why we added .transform()
                X_val_transformed = pca.transform(X_val)
                
                preds = fitted_model.predict(X_val_transformed)
            else:
                # --- This model used RAW data ---
                preds = fitted_model.predict(X_val)
                
            all_predictions.append(preds)
            all_weights.append(weight)

        predictions_stack = np.array(all_predictions)
        weights_array = np.array(all_weights)

        print("Calculating final weighted mode...")
        final_predictions = self._manual_weighted_mode(predictions_stack, weights_array)
        
        return final_predictions