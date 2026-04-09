import numpy as np

class LinTS:
    def __init__(self, n_features, alpha=0.5):
        """
        n_arms: Number of reinforcers (e.g., iPad, Chips, Praise -> 3)
        n_features: Number of context features (e.g., Bias, Emotion, Fatigue, Mastery, Preference -> 5)
        alpha: Exploration factor
        """

        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize parameters
        # B: Covariance matrices (n_arms, n_features, n_features) represents uncertainty
        # f: Reward history vectors (n_arms, n_features) represents accumulated signal
        # mu: Weight estimates (n_arms, n_features) represents the learned model weights
        self.B = np.identity(n_features)  
        self.f = np.zeros(n_features)
        self.mu = np.zeros(n_features)

    def select_arm(self, arm_context_matrix):
        """
        arm_context_matrix: Shape (n_arms, n_features)

        """
        n_arms = arm_context_matrix.shape[0]
        sampled_rewards = []

        B_inv = np.linalg.inv(self.B)
        self.mu = B_inv @ self.f
        
        theta_sample = np.random.multivariate_normal(
            self.mu, 
            (self.alpha**2) * B_inv
        )
        
        for i in range(n_arms):
            estimated_reward = np.dot(theta_sample, arm_context_matrix[i])
            sampled_rewards.append(estimated_reward)
            
        return np.argmax(sampled_rewards)
    
    def update(self, chosen_arm_context, reward):

        x = np.array(chosen_arm_context)
        
        # B += x * x^T
        self.B += np.outer(x, x)
        
        # f += x * reward
        self.f += x * reward
        

        # self.mu = np.linalg.inv(self.B) @ self.f

'''
    def update(self, chosen_arm, context_vector, reward):
        """
        Update model parameters based on the observed reward.
        
        chosen_arm: Index of the selected reinforcer
        context_vector: The feature vector used for the selection
        reward: The actual reward received (Emotion - Stickiness cost)
        """
        # Update the covariance matrix (uncertainty reduction)
        # B = B + x * x.T
        self.B[chosen_arm] += np.outer(context_vector, context_vector)
        
        # Update the reward history vector
        # f = f + x * r
        self.f[chosen_arm] += context_vector * reward
'''
'''
class LinTS:
    def __init__(self, n_arms, n_features, alpha=0.5):
        """
        n_arms: Number of reinforcers (e.g., iPad, Chips, Praise -> 3)
        n_features: Number of context features (e.g., Bias, Emotion, Fatigue, Mastery, Preference -> 5)
        alpha: Exploration factor
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize parameters
        # B: Covariance matrices (n_arms, n_features, n_features) represents uncertainty
        # f: Reward history vectors (n_arms, n_features) represents accumulated signal
        # mu: Weight estimates (n_arms, n_features) represents the learned model weights
        self.B = np.array([np.identity(n_features) for _ in range(n_arms)])
        self.f = np.array([np.zeros(n_features) for _ in range(n_arms)])
        self.mu = np.array([np.zeros(n_features) for _ in range(n_arms)])

    def select_arm(self, context_matrix):
        """
        Selects the best arm based on Thompson Sampling.
        
        context_matrix: Shape (n_arms, n_features)
        Note: Since each arm (reinforcer) has a specific preference value in the state, 
        the context vector seen by each arm is slightly different.
        """
        sampled_rewards = []
        
        for i in range(self.n_arms):
            # 1. Calculate the mean and variance of the posterior distribution
            # theta_hat = inv(B) * f [Mean]
            B_inv = np.linalg.inv(self.B[i]) # inverse matrix
            self.mu[i] = B_inv @ self.f[i] # calculate the mean
            
            # 2. Thompson Sampling: Sample theta from a multivariate normal distribution
            # variance = alpha^2 * inv(B) less alpha, more conservation 
            # This step adds randomness to explore uncertain arms
            theta_sample = np.random.multivariate_normal(
                self.mu[i], 
                (self.alpha**2) * B_inv
            )
            
            # 3. Estimate the Reward for this Arm
            # context_matrix[i] is the feature vector specific to the i-th reinforcer, linear
            estimated_reward = np.dot(theta_sample, context_matrix[i]) # dot multiply and calculation of exptected reward
            sampled_rewards.append(estimated_reward)
            
        # Return the index of the arm with the highest expected reward, index
        return np.argmax(sampled_rewards)
'''
