import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class EmbeddingAccumulator:
    def __init__(self, save_path='embeddings.npy', target_samples=1000):
        self.save_path = save_path
        self.target_samples = target_samples
        self.current_count = 0
        
    def add_batch(self, embeddings):
        """
        Add a batch of embeddings to the accumulator.
        
        Args:
            embeddings: numpy array of shape (batch_size, embedding_dim)
        """
        
        if os.path.exists(self.save_path):
            # Load existing embeddings
            existing_embeddings = np.load(self.save_path)
            # Concatenate with new batch
            all_embeddings = np.concatenate([existing_embeddings, embeddings], axis=0)
        else:
            all_embeddings = embeddings
            
        # Update count
        self.current_count = all_embeddings.shape[0]
        
        # Save updated embeddings
        np.save(self.save_path, all_embeddings)
        
        # print(f"Saved {self.current_count}/{self.target_samples} samples")
        
        # Check if we've reached target
        if self.current_count >= self.target_samples:
            return self.compute_statistics()
        return None
    
    def compute_statistics(self):
        """
        Compute per-dimension variance and average pairwise cosine similarity
        on a random 1k subset.
        """
        # Load all embeddings
        all_embeddings = np.load(self.save_path)
        
        # Sample 1000 random embeddings (or all if less than 1000)
        n_samples = min(self.target_samples, all_embeddings.shape[0])
        if all_embeddings.shape[0] > n_samples:
            indices = np.random.choice(all_embeddings.shape[0], n_samples, replace=False)
            sample_embeddings = all_embeddings[indices]
        else:
            sample_embeddings = all_embeddings
            
        print(f"Computing statistics on {sample_embeddings.shape[0]} samples")
        
        # Compute per-dimension variance
        per_dim_variance = np.var(sample_embeddings, axis=0)
        
        # Compute pairwise cosine similarities
        cos_sim_matrix = cosine_similarity(sample_embeddings)
        
        # Get upper triangle (excluding diagonal) for average
        mask = np.triu(np.ones_like(cos_sim_matrix, dtype=bool), k=1)
        pairwise_similarities = cos_sim_matrix[mask]
        avg_cosine_similarity = np.mean(pairwise_similarities)
        
        results = {
            'per_dimension_variance': per_dim_variance,
            'mean_variance': np.mean(per_dim_variance),
            'std_variance': np.std(per_dim_variance),
            'average_pairwise_cosine_similarity': avg_cosine_similarity,
            'n_samples_used': sample_embeddings.shape[0],
            'embedding_dimension': sample_embeddings.shape[1]
        }
        
        # Clean up the temporary file
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        
        print(results)
        return results

# Usage example:
def example_usage():
    """
    Example of how to use the accumulator with your model pipeline
    """
    accumulator = EmbeddingAccumulator()
    
    # Simulate processing batches through your model
    for batch_idx in range(10):  # Example: 10 batches
        # Your model processing here
        # model_output = your_model(batch_data)
        
        # Simulated embeddings (replace with actual model output)
        batch_embeddings = np.random.randn(100, 768)  # 100 samples, 768-dim embeddings
        
        # Add batch to accumulator
        results = accumulator.add_batch(batch_embeddings)
        
        # Check if we have results
        if results is not None:
            print("\n=== FINAL STATISTICS ===")
            print(f"Embedding dimension: {results['embedding_dimension']}")
            print(f"Samples used: {results['n_samples_used']}")
            print(f"Mean per-dimension variance: {results['mean_variance']:.4f}")
            print(f"Std per-dimension variance: {results['std_variance']:.4f}")
            print(f"Average pairwise cosine similarity: {results['average_pairwise_cosine_similarity']:.4f}")
            break

# Alternative: Simple function-based approach
def process_embedding_batch(embeddings, save_path='embeddings.npy', target_samples=1000):
    """
    Simple function to accumulate embeddings and compute stats when ready.
    
    Args:
        embeddings: numpy array of shape (batch_size, embedding_dim)
        save_path: path to save accumulated embeddings
        target_samples: number of samples to collect before computing stats
    
    Returns:
        None if not ready, dict with statistics if ready
    """
    # Load existing or create new
    if os.path.exists(save_path):
        existing = np.load(save_path)
        all_embs = np.concatenate([existing, embeddings], axis=0)
    else:
        all_embs = embeddings
    
    # Save updated
    np.save(save_path, all_embs)
    
    if all_embs.shape[0] >= target_samples:
        # Sample and compute stats
        indices = np.random.choice(all_embs.shape[0], target_samples, replace=False)
        sample = all_embs[indices]
        
        # Statistics
        per_dim_var = np.var(sample, axis=0)
        cos_sim_matrix = cosine_similarity(sample)
        mask = np.triu(np.ones_like(cos_sim_matrix, dtype=bool), k=1)
        avg_cos_sim = np.mean(cos_sim_matrix[mask])
        
        # Cleanup
        os.remove(save_path)
        
        return {
            'per_dimension_variance': per_dim_var,
            'average_pairwise_cosine_similarity': avg_cos_sim,
            'n_samples': len(sample)
        }
    
    return None