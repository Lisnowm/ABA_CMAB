import numpy as np


class VirtualChild:
    def __init__(self, profile_type, reinforcers_config):
        """
        reinforcers_config: list of dict, containing initial parameters for each reinforcer
        Example: [{'name': 'iPad', 'init_pref': 0.9, 'satiation': 0.8, 'recovery': 0.1}, ...]
        """

        """
        profile_type: str ('novelty_seeker', 'low_endurance', 'rigid')
        base_reinforcers: list of dicts (default config)
        """
        # self.reinforcers = reinforcers_config
        self.num_reinforcers = len(reinforcers_config)
        
        # Initialize dynamic states & multi initialization
        self.emotion = 0.0  # Range: -1 to 1 related to pref, emotion and fatigue
        self.fatigue = 0.0  # Range: 0 to 1, with time increasing
        self.focus = 0.0 # Range: 0 to 1, related to pref and fatigue

        # for novel seeker and rigid
        self.fatigue_base_cost = 0.05 
        self.fatigue_alpha = 2.0    


          # --- Apply Profile-Specific Logic to Parameters ---
        
        # Deep copy to avoid modifying the global config
        self.reinforcers = [r.copy() for r in reinforcers_config]

        if profile_type == 'novelty_seeker': 
            
            # 2. Detailed settings per reinforcer
            for r in self.reinforcers:

                if r['name'] == 'iPad':
                    # original: 0.7; 0.1
                    r['satiation_rate'] = 0.40  
                    r['recovery_rate'] = 0.90
                    r['init_pref'] = 0.95
                    r['transition'] = 0.45

                elif r['name'] == 'Chips':
                    # original: 0.4;0.3
                    r['satiation_rate'] = 0.30   
                    r['recovery_rate'] = 0.70
                    r['init_pref'] = 0.90
                    r['transition'] = 0.25

                elif r['name'] == 'Sticker':
                    # original: 0.4;0.3
                    r['satiation_rate'] = 0.20   
                    r['recovery_rate'] = 0.50
                    r['transition'] = 0.15
                    r['init_pref'] = 0.70
                       

        elif profile_type == 'low_endurance':
            self.fatigue_base_cost = 0.08 
            self.fatigue_alpha = 3.0      

        elif profile_type == 'rigid':
            for r in self.reinforcers:
                
                if r['name'] == 'iPad':
                    # preference item
                    r['init_pref'] = 0.9 # like most
                    r['satiation_rate'] = 0.9 # hard to get tired
                    r['recovery_rate'] = 0.7 # 
                    r['transition'] = 0.4
                
                elif r['name'] == 'Chips':
                    r['init_pref'] = 0.70
                    r['satiation_rate'] = 0.5
                    r['recovery_rate'] = 0.5
                    r['transition'] = 0.2
                    
                elif r['name'] == 'Sticker':
                    r['init_pref'] = 0.5
                    r['satiation_rate'] = 0.3  
                    r['recovery_rate'] = 0.2
                    r['transition'] = 0.1

        # For testing algorithm
        elif  profile_type == 'normal':
            pass
        
        self.current_prefs = [r['init_pref'] for r in self.reinforcers]

        
        # Record the last compliance status for subsequent analysis
        # self.last_compliance = 0

    def get_state(self):
        return {
            'emotion': self.emotion,
            'fatigue': self.fatigue,
            'preferences': self.current_prefs.copy(),
            'focus': self.focus
        }

    def react(self, task_difficulty, chosen_reinforcer_idx):
        """
        Generate Emotion and Compliance based on the task and selected reinforcer
        """
        pref = self.current_prefs[chosen_reinforcer_idx]
        reinforcer_data = self.reinforcers[chosen_reinforcer_idx]
        
        # After showing the reinforcer
        # --- 1. Focus Generation related to emotion and fatigue (Linear Formula) ---
        # Focus related to preference and fatugue
        noise_f = np.random.normal(-0.05, 0.05)
        w_f1, w_f2 = 0.8, 0.5
        raw_focus = (w_f1 * pref) - (w_f2 * self.fatigue) + noise_f
        
        # Clip to [0, 1], we try to avoid that
        self.focus = np.clip(raw_focus, 0.0, 1.0)

        # during the tasks
        # --- 2. Compliance Generation (Sigmoid) ---
        # Compliance is affected by: Emotion (+), Task Difficulty (-), Fatigue (-)
        # Logit = 2*Emotion - 2*Difficulty - 1*Fatigue + 1 (Bias)
        w_c1 = 0.2
        w_c2 = 0.3
        w_c3 = 0.5
        w_c4 = 0.6
        compliance_noise = np.random.normal(0, 0.05)
        raw_compliance = (w_c1 * self.emotion) - (w_c2 * task_difficulty) - (w_c3 * self.fatigue) + (w_c4 * self.focus) +compliance_noise
        compliance = np.clip(raw_compliance, -1.0, 1.0)

        # Discretion Compliance (for reporting / logging)
        discrete_compliance = self.discretize_5point(compliance)

        # finish the reinforcers
        # generalization of emotion change
        # E_t = w1*Pref + w2*E_prev - w3*Fatigue + Noise
        # w2 = 0.3, emotion changes a lot with the preference and task, suitable for ASD kids
        w1, w2, w3 = 0.6, 0.3, 0.4
        noise = np.random.normal(0, 0.05)
        raw_emotion = (w1 * pref) + (w2 * self.emotion) - (w3 * self.fatigue) + noise # showing the reinforcer instead of having that
        # Clip to [-1, 1], we try to avoid that
        self.emotion = np.clip(raw_emotion, -1.0, 1.0)
        

        # Discretize Emotion (for reporting/logging)
        discrete_emotion = self.discretize_5point(self.emotion)

        base_stickiness = reinforcer_data.get('transition', 0.2)
        
        w_t1 = 1.5   
        w_t2 = 0.5
        w_t3 = 0.3
        t_noise = np.random.normal(0, 0.05)
        raw_resistance = (w_t1 * base_stickiness * (1 + pref)) + \
                         (w_t2 * self.fatigue) - \
                         (w_t3 * self.emotion) + \
                         t_noise
        
        resistance = np.clip(raw_resistance, -1.0, 1.0)
        discrete_resistance = self.discretize_5point(resistance)

        return discrete_emotion, discrete_compliance, discrete_resistance, self.emotion, compliance, resistance, self.focus, self.fatigue
    
    #    return self.emotion, compliance, self.focus
    
    def _calculate_internal_logic(self, task_difficulty, chosen_reinforcer_idx, current_state):
        """
        Internal helper: Calculates theoretical Focus, Compliance, and Emotion (New)
        WITHOUT noise. Strictly follows the logic order in 'react'.
        For regret calculation
        """
        pref = current_state['preferences'][chosen_reinforcer_idx]
        fatigue = current_state['fatigue']
        old_emotion = current_state['emotion'] # The emotion BEFORE the task]
        reinforcer_data = self.reinforcers[chosen_reinforcer_idx]

        # --- 1. Expected Focus (No Noise) ---
        # Logic matches react: w_f1 * pref - w_f2 * fatigue
        w_f1, w_f2 = 0.8, 0.5
        raw_focus = (w_f1 * pref) - (w_f2 * fatigue)
        exp_focus = np.clip(raw_focus, 0.0, 1.0)

        # --- 2. Expected Compliance (No Noise) ---
        # Logic matches react: uses OLD emotion, Difficulty, Fatigue, and NEW Focus
        w_c1, w_c2, w_c3, w_c4 = 0.4, 0.3, 0.5, 0.6
        raw_compliance = (w_c1 * old_emotion) - \
                         (w_c2 * task_difficulty) - \
                         (w_c3 * fatigue) + \
                         (w_c4 * exp_focus)
        exp_compliance = np.clip(raw_compliance, -1.0, 1.0)

        # --- 3. Expected New Emotion (No Noise) ---
        # Logic matches react: uses Pref, Old Emotion, Fatigue
        w1, w2, w3 = 0.6, 0.3, 0.4
        raw_new_emotion = (w1 * pref) + (w2 * old_emotion) - (w3 * fatigue)
        exp_new_emotion = np.clip(raw_new_emotion, -1.0, 1.0)

        # --- 4. Expected Resistance (No Noise)---
        base_stickiness = reinforcer_data.get('transition', 0.2)
        w_t1, w_t2, w_t3 = 1.5, 0.5, 0.3
        raw_resistance = (w_t1 * base_stickiness * (1 + pref)) + \
                         (w_t2 * fatigue) - \
                         (w_t3 * exp_new_emotion)
        exp_resistance = np.clip(raw_resistance, -1.0, 1.0)

        return exp_focus, exp_compliance, exp_new_emotion, exp_resistance

    def get_expected_reward(self, task_difficulty, chosen_reinforcer_idx, reinforcer_data):
        """
        Public method to calculate the Expected Reward for a specific arm.
        Used by the Main Loop to find the optimal arm.
        """
        # Snapshot current state
        current_state = {
            'preferences': self.current_prefs,
            'fatigue': self.fatigue,
            'emotion': self.emotion
        }

        # Get theoretical values (Deterministic)
        exp_focus, exp_comp, exp_new_emo, exp_resistance = self._calculate_internal_logic(
            task_difficulty, chosen_reinforcer_idx, current_state
        )
        # Define Weights (Must match what you care about in the simulation)
        W_FOC = 10.0
        W_COMPLIANCE = 8.0  # Added Compliance as discussed
        W_EMOTION = 5.0
        W_resistance = 10.0
        #W_transition = 3.0

        # Calculate Reward
        # Note: We use exp_new_emo (the outcome emotion)
        reward = (W_FOC * exp_focus) + \
                 (W_COMPLIANCE * exp_comp) + \
                 (W_EMOTION * exp_new_emo) - \
                 (W_resistance * exp_resistance)
        
        return reward

    def update_internal_states(self, chosen_idx,reinforcer_recovery_val):
        """
        Update after task completion: Satiation, Recovery, Fatigue
        """
        # 1. Update Preferences (Satiation & Recovery)
        for i in range(self.num_reinforcers):
            if i == chosen_idx:
                # Chosen: Satiation (Preference decreases)
                # Pref = Pref * satiation_rate (e.g., 0.9 * 0.8 = 0.72)
                rate = self.reinforcers[i]['satiation_rate']
                self.current_prefs[i] *= rate
            else:
                # Not Chosen: Recovery (Preference increases back to initial)
                # Pref = Current + (Init - Current) * recovery_rate
                init = self.reinforcers[i]['init_pref']
                curr = self.current_prefs[i]
                rec_rate = self.reinforcers[i]['recovery_rate']
                self.current_prefs[i] = curr + (init - curr) * rec_rate
        
        # Clip preferences to [0, 1] just in case
        self.current_prefs = np.clip(self.current_prefs, 0.0, 1.0)

        # 2. Update Fatigue (Linear increase)
        # Formula: Growth = Base * (1 + Alpha * Current_Fatigue)
        growth = self.fatigue_base_cost * (1 + self.fatigue_alpha * self.fatigue)
        # Net change = Growth - Recovery from item (e.g., Chips)
        net_change = growth - reinforcer_recovery_val
        self.fatigue += net_change
        self.fatigue = max(0.0, min(1.0, self.fatigue))

    def discretize_5point(self, val): 
            """
            For Emotion and Compliance
            """
            if val >= 0.6: return 1.0      # (Very Happy / Independent)
            elif val >= 0.2: return 0.5    # (Happy / Prompted)
            elif val >= -0.2: return 0.0   # (Normal / No Response)
            elif val >= -0.6: return -0.5  # (Unhappy / Avoidance)
            else: return -1.0              # (Very Unhappy / Refusal)