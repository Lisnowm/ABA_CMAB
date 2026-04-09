from new_LinearThompsonSampling import LinTS
import numpy as np

class VirtualTherapist:
    def __init__(self, n_arms, n_features):
        self.brain = LinTS(n_features=9) 
        self.n_arms = n_arms

    def choose_reinforcer(self, child_state, task_mastery, difficulty, REINFORCERS):
        n_features = self.brain.n_features
        
        context_matrix = np.zeros((self.n_arms, n_features))
    
        for i in range(self.n_arms):
            transition_cost = REINFORCERS[i]['transition']
            current_pref = child_state['preferences'][i]
            fatigue = child_state['fatigue']

            context_matrix[i] = [
                1.0,                # Bias
                child_state['emotion'], 
                fatigue, 
                task_mastery,
                difficulty,
                
                # Arm
                current_pref,
                transition_cost,
                
                # Interaction
                fatigue * transition_cost,
                child_state['emotion'] * current_pref
            ]
            
        chosen_idx = self.brain.select_arm(context_matrix)

        return chosen_idx, context_matrix[chosen_idx]
    
    def update_strategy(self,context_vector, reward):
        self.brain.update(context_vector, reward)

'''        for i in range(n_arms):
            # Feature 0: Bias
            # Feature 1: Child Emotion
            # Feature 2: Child Fatigue
            # Feature 3: Task Mastery
            # Feature 4: Difficulty
            # Feature 5: Preference for THIS specific arm (This is the key to distinguishing different Arms)
            # Feature 6: stickness
        
            context_matrix[i] = [
                1.0, 
                child_state['emotion'], 
                child_state['fatigue'], 
                task_mastery,
                difficulty,
                child_state['preferences'][i], 
                REINFORCERS[i]['stickiness']
            ]
'''
'''
    def choose_reinforcer(self, child_state, task_mastery,difficulty,REINFORCERS):
        """
        Construct Context Matrix and query the algorithm
        Context Features: [Bias, Emotion, Fatigue, Task_Mastery, Item_Preference]
        """
        n_arms = self.brain.n_arms
        n_features = self.brain.n_features
        
        # Construct Context Matrix (n_arms x n_features)
        # Each row represents the feature vector if that specific Arm is chosen
        context_matrix = np.zeros((n_arms, n_features))
    
        for i in range(n_arms):
            # Feature 0: Bias
            # Feature 1: Child Emotion
            # Feature 2: Child Fatigue
            # Feature 3: Task Mastery
            # Feature 4: Difficulty
            # Feature 5: Preference for THIS specific arm (This is the key to distinguishing different Arms)
            # Feature 6: stickness
        
            context_matrix[i] = [
                1.0, 
                child_state['emotion'], 
                child_state['fatigue'], 
                task_mastery,
                difficulty,
                child_state['preferences'][i], 
                #REINFORCERS[i]['transition']
                #child_state['preferences'][i] - REINFORCERS[i]['transition'] # Net Value (Pref - Cost)
                #child_state['preferences'][i] * (1 - child_state['fatigue']) # Adjusted Pref by Fatigue
            ]
            
        chosen_idx = self.brain.select_arm(context_matrix) # with the index of highest arm
        return chosen_idx, context_matrix[chosen_idx] # return the chosen index and context
'''